import json
import os
import shutil
import cv2
import torch
import asyncio
import aiohttp
from django.http import JsonResponse, HttpResponse
import numpy as np
from PIL import Image
import joblib
import torchvision.transforms as transforms
from transformers import AutoModelForImageClassification, AutoProcessor
from io import BytesIO
from datetime import datetime, timezone
from .webodmapi_class import AzureBlobClient  # Ensure you have implemented this class
import pymssql
import hashlib
from ultralytics import YOLO
import timm
import os
import logging
import warnings
# Suppress aiohttp warnings
warnings.filterwarnings("ignore", message="Unclosed client session")
warnings.filterwarnings("ignore", message="Unclosed connector")
# Or suppress asyncio warnings specifically
logger = logging.getLogger('collector')

# Load DinoV2 Model
dinov2_model = AutoModelForImageClassification.from_pretrained("cvmil/dinov2-base_rice-leaf-disease-augmented_fft")
dinov2_model.eval()
dinov2_processor = AutoProcessor.from_pretrained("cvmil/dinov2-base_rice-leaf-disease-augmented_fft")

# Load EfficientNet Model
EB3_MODEL_PATH = "/mnt/models/checkpoint_epoch_30.pth"
checkpoint = torch.load(EB3_MODEL_PATH, map_location="cpu")

efficientnet_model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=9)
efficientnet_model.load_state_dict(checkpoint["model_state_dict"]) 
efficientnet_model.eval()

SUGAR_MODEL_PATH = "/mnt/models/checkpoint_epoch_20_finetune.pth"
sugar_checkpoint = torch.load(SUGAR_MODEL_PATH, map_location="cpu")
efficientnet_sugar_model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=9)
efficientnet_sugar_model.load_state_dict(sugar_checkpoint["model_state_dict"])
efficientnet_sugar_model.eval()

# Load Sugarcane Pest Model
SUGAR_PEST_MODEL_PATH = "/mnt/models/sugarcane_pest_checkpoint_epoch_30.pth"
sugar_pest_checkpoint = torch.load(SUGAR_PEST_MODEL_PATH, map_location="cpu")
efficientnet_sugar_pest_model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=6)
efficientnet_sugar_pest_model.load_state_dict(sugar_pest_checkpoint["model_state_dict"])
efficientnet_sugar_pest_model.eval()

# Load class mapping for pest
sugar_pest_class_map = joblib.load("/mnt/models/sugarcane_pest_class_map.pkl")
sugar_pest_index_to_label = {v: k for k, v in sugar_pest_class_map.items()}


# Load Random Forest Meta Classifier
rf_model = joblib.load("/mnt/models/meta_model.pkl")

# Load Class Mapping
class_map = joblib.load("/mnt/models/class_map.pkl")  # Ensure this file contains a dictionary mapping indices to class names

# Image Transformation
efficientnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def generate_image_hash(image_data):
    """Generate a SHA256 hash of the image data."""
    return hashlib.sha256(image_data).hexdigest()

def get_predictions(image):
    """Extract features from both models and stack them."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # DinoV2 Feature Extraction
    inputs = dinov2_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        dinov2_logits = dinov2_model(**inputs).logits.cpu().numpy()
    
    # EfficientNet Feature Extraction
    image_tensor = efficientnet_transform(image).unsqueeze(0)
    with torch.no_grad():
        efficientnet_preds = efficientnet_model(image_tensor).cpu().numpy()
    
    # Stack Features
    stacked_features = np.hstack([dinov2_logits, efficientnet_preds])
    return stacked_features


def predict_disease(image):
    """Predict disease using stacked features with a threshold of 0.5."""
    features = get_predictions(image)
    
    # Get probability estimates
    probabilities = rf_model.predict_proba(features)
    max_prob = np.max(probabilities)
    
    if max_prob < 0.7:
        return "Unknown Disease"
    
    predicted_class = np.argmax(probabilities)
    
    logger.info(f"Predicted Raw Class: {predicted_class}")  # Debugging
    logger.info(f"Available Class Indices: {list(class_map.values())}")  # Debugging
    
    if predicted_class not in class_map.values():
        return f"Unknown Class {predicted_class}"
    
    # Reverse lookup class name
    predicted_label = [k for k, v in class_map.items() if v == predicted_class]
    return predicted_label[0] if predicted_label else f"Unknown Class {predicted_class}"

def predict_sugar_disease(image):
    """Predict disease directly from EfficientNetB3."""
    class_map = joblib.load("/mnt/models/sugarcane_class_map_new.pkl")
    index_to_label = {v: k for k, v in class_map.items()}
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    image_tensor = efficientnet_transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs =  efficientnet_sugar_model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
    
    predicted_class = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))
    
    if predicted_class not in index_to_label:
        return f"Unknown Class {predicted_class}"
    
    label = index_to_label[predicted_class]
    return f"{label}"

def predict_sugar_pest(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    image_tensor = efficientnet_transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = efficientnet_sugar_pest_model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
    
    predicted_class = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))

    if predicted_class not in sugar_pest_index_to_label:
        return f"Unknown Class {predicted_class}"
    
    label = sugar_pest_index_to_label[predicted_class]
    return f"{label}"


async def process_images(request):
    """Django view to process multiple images and upload results to Azure Blob Storage."""
    if request.method != "POST":
        return JsonResponse({"error": "Only POST method is allowed."}, status=405)

    farm_id = request.POST.get("farm_id")
    shoot_date = request.POST.get("shoot_date")
    crop_name = request.POST.get("crop_name")
    
    image_files = request.FILES.getlist("images")
    
    if not image_files:
        return JsonResponse({"error": "No image files provided."}, status=400)

    repo_root = os.path.dirname(os.path.abspath(__file__))

    if not os.path.exists(EB3_MODEL_PATH):
        return JsonResponse({"error": "Config or weight file not found."}, status=500)
    
    try:
        connection = pymssql.connect(
            server=os.environ.get('DB_HOST') + ":" + os.environ.get('DB_PORT'),
            user=os.environ.get('SQL_USER'),
            password=os.environ.get('SQL_PASSWORD'),
            database=os.environ.get('SQL_DB')   
        )
        logger.info("Database connection established.")
       
        
        
        connection_string = (
            f"DefaultEndpointsProtocol=https;"
            f"AccountName={os.environ.get('AZURE_SA_NAME')};"
            f"AccountKey={os.environ.get('AZURE_SA_ACCESSKEY')};"
            f"EndpointSuffix=core.windows.net"
        )
        
        azure_blob_client = AzureBlobClient(connection_string, "disease-detection-cnn", os.environ.get('AZURE_SA_ACCESSKEY'))
        await azure_blob_client.ensure_container_exists()
        
        output_dir = os.path.join(os.getenv("XDG_CONFIG_HOME"), ".cache", "farmvibes-ai", "disease-detection-cnn")
       
        os.makedirs(output_dir, exist_ok=True)
        
        
        current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        current_time = datetime.now(timezone.utc).strftime("%H-%M-%S") + "UTC"

        processed_images = []
        
        with connection.cursor() as cursor:
            
            for image_index, image_file in enumerate(image_files):  # Use image_index to ensure uniqueness
                
                image_data = image_file.read()
                
                image_hash = generate_image_hash(image_data)
                logger.info(image_hash)
                logger.info(image_index)
            
                # Check if the image hash already exists in the database
                cursor.execute("SELECT PredictedDiseaseName,CaptureImageUrl FROM AMDiseaseDetectionCnn WHERE ImageHash = %s AND MasterFarmId = %s AND ShootDate = %s AND CropName = %s",(image_hash, farm_id,shoot_date,crop_name))
                
                existing_record = cursor.fetchone()

                if existing_record:
                    logger.info(f"Image already exists: {existing_record}")
                    query = f"""UPDATE dbo.AMDiseaseDetectionCnn
                            SET
                                IsWorkFlowStatus = 2
                            WHERE
                            MasterFarmId = {farm_id}
                            AND ShootDate='{shoot_date}'
                            AND ImageHash='{image_hash}'
                            AND CropName = '{crop_name}'
                            """
                    cursor.execute(query)
                    connection.commit()
                    continue  # Skip processing for duplicate images

                
                query = f"""
                        INSERT INTO dbo.AMDiseaseDetectionCnn (
                        MasterFarmId,ShootDate,ImageHash,IsWorkFlowStatus,EnterDate,CropName
                        )
                        VALUES ({farm_id},'{shoot_date}','{image_hash}',1,'{datetime.now()}','{crop_name}');
                 
                        """
                cursor.execute(query)
                connection.commit()
                
                image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(image_rgb)
                
                if crop_name=="Rice":

                    prediction = predict_disease(image_pil)

                if crop_name=="Sugarcane":
                    prediction = predict_sugar_disease(image_pil)

                if crop_name=="Sugarcane Pest":
                    prediction = predict_sugar_pest(image_pil)

                logger.info(prediction)
               
                

                # Save images locally before uploading
                local_input_image_path = os.path.join(output_dir, f"input_{farm_id}_{shoot_date}_{current_date}_{current_time}_{image_index}.jpg")
                

                cv2.imwrite(local_input_image_path,image)
                
                
                
                # Upload images to Azure Blob Storage
                input_blob_name = f"{farm_id}/{shoot_date}/input_{crop_name}_{farm_id}_{shoot_date}_{current_date}_{current_time}_{image_index}.jpg"
                

                await azure_blob_client.upload_blob(local_input_image_path, input_blob_name)
                

                input_blob_url = azure_blob_client.get_blob_url(input_blob_name)
                
                
                logger.info(f"Uploading {image_file.name} -> Input URL: {input_blob_url}, PredictedDiseaseName : {prediction}")
                
                modified_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                query = f"""
                    UPDATE dbo.AMDiseaseDetectionCnn
                    SET PredictedDiseaseName = '{prediction}', 
                        CaptureImageUrl = '{input_blob_url}', 
                        IsWorkFlowStatus = 2, 
                        ModifiedById = 1, 
                        ModifiedDate = '{modified_date}',
                        DiseaseDetectionCnnDate='{modified_date}',
                        IsActive=1

                    WHERE MasterFarmId = {farm_id} AND ShootDate = '{shoot_date}' AND ImageHash = '{image_hash}' AND CropName = '{crop_name}'
                """

                # Execute the query with parameters
                cursor.execute(query)

                connection.commit()
                
                processed_images.append({
                    "input_image_url": input_blob_url,
                    "PredictedDiseaseName": prediction
                })

        return JsonResponse({
            "message": "Disease has been predicted for the uploaded images successfully.",
            "processed_images": processed_images
        }, status=200)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


    finally:
        logger.info("Starting cleanup process")
        
        # Clean up temporary directory
        if output_dir and os.path.exists(output_dir):
            try:
                shutil.rmtree(output_dir)
                logger.info(f"Directory '{output_dir}' has been deleted.")
            except Exception as e:
                logger.warning(f"Could not delete directory {output_dir}: {str(e)}")
        
        # Close database connection
        if connection:
            try:
                connection.close()
                logger.info("Database connection closed.")
            except Exception as e:
                logger.warning(f"Error closing database connection: {str(e)}")