import streamlit as st
import torch
import numpy as np
from PIL import Image
import joblib
import torchvision.transforms as transforms
from transformers import AutoModelForImageClassification, AutoImageProcessor
import timm
import os

# Load Swin Model
swin_model = AutoModelForImageClassification.from_pretrained("rmezapi/sugarcane-diagnosis-swin-tiny")
swin_model.eval()
swin_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

# Load EfficientNet Model
EB3_MODEL_PATH = "./checkpoint_epoch_30.pth"
checkpoint = torch.load(EB3_MODEL_PATH, map_location="cpu")

efficientnet_model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=8)
efficientnet_model.load_state_dict(checkpoint["model_state_dict"]) 
efficientnet_model.eval()

# Load Random Forest Meta Classifier
rf_model = joblib.load("./sugarcane_meta_model.pkl")

# Load Class Mapping
class_map = joblib.load("./sugarcane_class_map.pkl")  # Ensure this file contains a dictionary mapping indices to class names

# Image Transformation
efficientnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_predictions(image):
    """Extract features from both models and stack them."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # swin Feature Extraction
    inputs = swin_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        swin_logits = swin_model(**inputs).logits.cpu().numpy()
    
    # EfficientNet Feature Extraction
    image_tensor = efficientnet_transform(image).unsqueeze(0)
    with torch.no_grad():
        efficientnet_preds = efficientnet_model(image_tensor).cpu().numpy()
    
    # Stack Features
    stacked_features = np.hstack([swin_logits, efficientnet_preds])
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
    
    print(f"Predicted Raw Class: {predicted_class}")  # Debugging
    print(f"Available Class Indices: {list(class_map.values())}")  # Debugging
    
    if predicted_class not in class_map.values():
        return f"Unknown Class {predicted_class}"
    
    # Reverse lookup class name
    predicted_label = [k for k, v in class_map.items() if v == predicted_class]
    return predicted_label[0] if predicted_label else f"Unknown Class {predicted_class}"


# Streamlit UI
st.title("🌾 Rice Disease Classifier")
st.write("Upload an image of a rice leaf to detect the disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg", "bmp", "tiff", "gif", "webp"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Predict Disease"):
        with st.spinner("Analyzing..."):
            prediction = predict_disease(image)
        st.success(f"Predicted Disease: **{prediction}**")
