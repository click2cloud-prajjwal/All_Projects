import torch
import torchvision.transforms as transforms
from PIL import Image
import streamlit as st
from wrapper import EnsembleModel, model_paths

model_weights = [0.8, 0.5, 0.3]

# Load model with custom weights
ensemble_model = EnsembleModel(model_paths, model_weights=model_weights)
ensemble_model.load_state_dict(torch.load("/home/ubuntu/yolo_disease/ensemble_model_853.pth", map_location="cpu"))
ensemble_model.eval()

# Confidence threshold and temperature scaling
THRESHOLD = 0.5
TEMPERATURE = 0.25

# Define class labels
class_labels = {
    0: "Bacterial Leaf Blight",
    1: "Brown Spot",
    2: "Healthy",
    3: "Leaf Blast",
    4: "Leaf Scald",
    5: "Narrow Brown Spot",
    6: "Neck Blast",
    7: "Sheath Blight",
    8: "Tungro"
}

# Preprocess function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image).convert("RGB")
    return transform(image).unsqueeze(0)

# Streamlit UI
st.title("Rice Disease Classification")
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    image = preprocess_image(uploaded_file)

    if st.button("Predict"):
        with torch.no_grad():
            logits = ensemble_model(image)
            probs = torch.softmax(logits / TEMPERATURE, dim=1)  # Apply temperature scaling
            predicted_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, predicted_class].item()

        if confidence >= THRESHOLD:
            predicted_label = class_labels.get(predicted_class, "Unknown")
        else:
            predicted_label = "Unknown Disease"

        st.write(f"**Prediction:** {predicted_label}")
        st.write(f"**Confidence:** {confidence:.2f}")