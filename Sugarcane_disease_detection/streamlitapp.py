import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import timm
import json

# Load class index to name mapping
idx_to_class = {
    0: "Grassy shoot",
    1: "Healthy",
    2: "Mosaic",
    3: "Pokkah Boeng",
    4: "RedRot",
    5: "Rust",
    6: "Yellow",
    7: "Smut"
}

# Load the trained model
@st.cache_resource
def load_model():
    model = timm.create_model('efficientnet_b3', pretrained=False, num_classes=len(idx_to_class))
    model.load_state_dict(torch.load("checkpoint_epoch_25.pth", map_location=torch.device("cpu"))["model_state_dict"])
    model.eval()
    return model

# Preprocessing pipeline
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # standard ImageNet normalization
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Prediction function
def predict(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
    return idx_to_class[predicted_class.item()], confidence.item()

# Streamlit UI
st.title("🌾 Sugarcane Disease Detection - EfficientNetB3")
st.write("Upload a sugarcane leaf image to detect the disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with st.spinner("Analyzing image..."):
        model = load_model()
        image_tensor = preprocess_image(image)
        label, confidence = predict(model, image_tensor)
    
    st.success(f"✅ Predicted: **{label}**")
    st.info(f"🔍 Confidence: **{confidence*100:.2f}%**")
    
    # Display disease information
    disease_info = {
        "Grassy shoot": "A disease characterized by thin, grassy tillers. Caused by phytoplasma.",
        "Healthy": "No disease detected. The plant appears to be in good condition.",
        "Mosaic": "Characterized by yellowish to light green patches. Caused by Sugarcane Mosaic Virus.",
        "Pokkah Boeng": "Shows wrinkled and twisted young leaves. Caused by Fusarium moniliforme.",
        "RedRot": "Shows red discoloration in the stem tissues. Caused by Colletotrichum falcatum.",
        "Rust": "Shows orange-brown or reddish-brown pustules on leaves. Caused by Puccinia melanocephala.",
        "Yellow": "Shows uniform yellowing of leaves. Often caused by nutritional deficiencies or Yellow Leaf Virus.",
        "Smut": "Characterized by black whip-like structures. Caused by Sporisorium scitamineum."
    }
    
    st.subheader("Disease Information")
    st.write(disease_info[label])
    
    # Show all class probabilities
    if st.checkbox("Show detailed analysis"):
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
        
        probs_dict = {idx_to_class[i]: float(probabilities[i]) * 100 for i in range(len(idx_to_class))}
        sorted_probs = dict(sorted(probs_dict.items(), key=lambda x: x[1], reverse=True))
        
        st.subheader("Class Probabilities")
        for disease, prob in sorted_probs.items():
            st.write(f"{disease}: {prob:.2f}%")