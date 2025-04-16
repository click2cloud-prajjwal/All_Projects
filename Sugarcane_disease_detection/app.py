import streamlit as st
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import timm
import joblib

# Load EfficientNetB3 Model
EB3_MODEL_PATH = "/home/ubuntu/sugarcane_disease_detection/sugarcane_disease/checkpoint_epoch_25.pth"
checkpoint = torch.load(EB3_MODEL_PATH, map_location="cpu")

efficientnet_model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=8)
efficientnet_model.load_state_dict(checkpoint["model_state_dict"])
efficientnet_model.eval()

# Load Class Mapping
class_map = joblib.load("./sugarcane_class_map.pkl")  # dict like {'Red Rot': 0, 'Healthy': 1, ...}

# Reverse class_map to make index -> label
index_to_label = {v: k for k, v in class_map.items()}

# Image Transformation
efficientnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_disease(image):
    """Predict disease directly from EfficientNetB3."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    image_tensor = efficientnet_transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = efficientnet_model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
    
    predicted_class = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))
    
    if predicted_class not in index_to_label:
        return f"Unknown Class {predicted_class}"
    
    label = index_to_label[predicted_class]
    return f"{label} ({confidence * 100:.2f}% confidence)"

# Streamlit UI
st.title("🌾 Sugarcane Disease Classifier")
st.write("Upload an image of a sugarcane leaf to detect the disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg", "bmp", "tiff", "gif", "webp"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Predict Disease"):
        with st.spinner("Analyzing..."):
            prediction = predict_disease(image)
        st.success(f"Predicted Disease: **{prediction}**")
