import os
import torch
import hashlib
from datetime import datetime, timezone
from azure.storage.blob import BlobServiceClient, PublicAccess
from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from safetensors.torch import load_file
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

app = Flask(__name__)

class SegmentAnythingService:
    def __init__(self):
        self.device = "cpu"
        self.model_container = "segmentanythingmodels"
        self.sam_filename = "sam_vit_b_01ec64.pth"
        self.lora_filename = "lora_rank512.safetensors"
        self.local_model_dir = "./models"
        os.makedirs(self.local_model_dir, exist_ok=True)
        self.sam_path = os.path.join(self.local_model_dir, self.sam_filename)
        self.lora_path = os.path.join(self.local_model_dir, self.lora_filename)

        self._download_model_files()
        self._load_models()

    def _download_model_files(self):
        connection_str = (
            f"DefaultEndpointsProtocol=https;"
            f"AccountName={os.getenv('AZURE_SA_NAME')};"
            f"AccountKey={os.getenv('AZURE_SA_ACCESSKEY')};"
            f"EndpointSuffix=core.windows.net"
        )
        blob_service = BlobServiceClient.from_connection_string(connection_str)
        container_client = blob_service.get_container_client(self.model_container)

        for blob_name, file_path in [(self.sam_filename, self.sam_path), (self.lora_filename, self.lora_path)]:
            blob_client = container_client.get_blob_client(blob_name)
            if not os.path.exists(file_path):
                with open(file_path, "wb") as f:
                    f.write(blob_client.download_blob().readall())

    def _load_models(self):
        self.sam = sam_model_registry["vit_b"](checkpoint=self.sam_path)
        self.sam.load_state_dict(load_file(self.lora_path), strict=False)
        self.sam.eval().to(self.device)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)

    def generate_image_hash(self, image_bytes):
        return hashlib.sha256(image_bytes).hexdigest()

    def show_anns(self, anns):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=lambda x: x["area"], reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)
        for ann in sorted_anns:
            m = ann["segmentation"]
            img = np.ones((m.shape[0], m.shape[1], 3))
            color_mask = np.random.random((1, 3)).tolist()[0]
            for i in range(3):
                img[:, :, i] = color_mask[i]
            ax.imshow(np.dstack((img, m * 0.35)))

    def overlay_mask(self, image, masks):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        self.show_anns(masks)
        plt.axis("off")
        buffer = BytesIO()
        plt.savefig(buffer, format='PNG', bbox_inches='tight', pad_inches=0)
        plt.close()
        buffer.seek(0)
        return buffer

    def preprocess_image(self, img: Image.Image) -> np.ndarray:
        # Resize image if needed (optional step, SAM usually works with high res)
        max_dim = 1024
        if max(img.size) > max_dim:
            scale = max_dim / max(img.size)
            new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        # Convert to numpy and normalize
        img_array = np.array(img).astype(np.float32)

        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        img_array = (img_array - mean) / std


        return img_array

    def process_image(self, image_bytes):
        image_hash = self.generate_image_hash(image_bytes)
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        preprocessed_array = self.preprocess_image(img)

        # Use raw image for visualization, preprocessed only for mask generation
        masks = self.mask_generator.generate(preprocessed_array)

        segmented_buffer = self.overlay_mask(np.array(img), masks)

        # Upload to blob
        connection_str = (
            f"DefaultEndpointsProtocol=https;"
            f"AccountName={os.getenv('AZURE_SA_NAME')};"
            f"AccountKey={os.getenv('AZURE_SA_ACCESSKEY')};"
            f"EndpointSuffix=core.windows.net"
        )
        blob_service = BlobServiceClient.from_connection_string(connection_str)
        container_name = "segmentanythingtest"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        file_name = f"segmented_{image_hash}_{timestamp}.png"
        container_client = blob_service.get_container_client(container_name)

        if not container_client.exists():
            container_client.create_container(public_access=PublicAccess.Blob)

        blob_client = container_client.get_blob_client(file_name)
        blob_client.upload_blob(segmented_buffer.getvalue(), blob_type="BlockBlob", overwrite=False)

        return blob_client.url

segment_service = SegmentAnythingService()

@app.route("/segment", methods=["POST"])
def segment():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    image_file = request.files["image"]
    try:
        image_bytes = image_file.read()
        segmented_url = segment_service.process_image(image_bytes)
        return jsonify({"message": "Success", "segmented_image_url": segmented_url})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=8000)
