import torch
from flask import Flask, request, jsonify
from PIL import Image
from torchvision import transforms
from transformers import SwinForImageClassification
import io
import torch.nn.functional as F

# Initialize Flask app
app = Flask(__name__)

# Load the Swin model and processor
model_path = "/home/ubuntu/sugarcane_disease_detection/model_chkpt/swin_tiny_epoch10.pth"
model = SwinForImageClassification.from_pretrained('rmezapi/sugarcane-diagnosis-swin-tiny')
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
model.eval()

# Define the class names
class_names = [
    "BrownRust",
    "Dried Leaves",
    "Grassy shoot",
    "Healthy",
    "Mosaic",
    "Pokkah Boeng",
    "Red Rot",
    "Rust",
    "Smut",
    "Yellow"
]

# Image pre-processing transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.75

# Define the route for image classification
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read and preprocess the image
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        image = transform(image).unsqueeze(0)  # Add batch dimension

        # Inference
        with torch.no_grad():
            outputs = model(image)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
            confidence, predicted_class_idx = torch.max(probs, dim=1)
            confidence = confidence.item()

            if confidence < CONFIDENCE_THRESHOLD:
                predicted_class_name = "Unknown Disease"
                predicted_class_idx = -1  # Optional: you can also omit this field
            else:
                predicted_class_name = class_names[predicted_class_idx.item()]
                predicted_class_idx = predicted_class_idx.item()

        return jsonify({
            'predicted_class_idx': predicted_class_idx,
            'predicted_class_name': predicted_class_name,
            'confidence': round(confidence, 4)
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
