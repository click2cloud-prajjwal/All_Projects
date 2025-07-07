import streamlit as st
import torch
import torch.nn.functional as F
import timm
from torchvision import transforms
from PIL import Image

# Load model using timm
@st.cache_resource
def load_model(weight_path, num_classes):
    model = timm.create_model('efficientnet_b3', pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Set parameters
NUM_CLASSES = 6  # Change this to match your problem
WEIGHT_PATH = "/home/ubuntu/pests_and_disease/Rice_pest_detection/Rice_pest_classifier.pth"  # Update with your actual path
CONFIDENCE_THRESHOLD = 0.5  # 50% confidence threshold

# Load the model
model = load_model(WEIGHT_PATH, NUM_CLASSES)

# Define your transforms (same as during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Streamlit UI
st.title("Rice Pest Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg","webp"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)
        
        confidence_score = confidence.item()
        predicted_class_idx = predicted_class.item()

    # Class names mapping
    class_names = ['Borers', 'Grasshopper', 'Hopper', 'Mealybug', 'Rice Leaf Roller', 'Thrips']

    # Display results based on confidence threshold
    st.subheader("Prediction Results")
    
    if confidence_score >= CONFIDENCE_THRESHOLD:
        st.success(f"**Prediction:** {class_names[predicted_class_idx]}")
        st.info(f"**Confidence Score:** {confidence_score:.2%}")
    else:
        st.warning(f"**Low Confidence Prediction:** {class_names[predicted_class_idx]}")
        st.warning(f"**Confidence Score:** {confidence_score:.2%} (Below {CONFIDENCE_THRESHOLD:.0%} threshold)")
        st.error("⚠️ The model is not confident about this prediction. Please try with a clearer image or consult an expert.")
    
    # Show all class probabilities
    st.subheader("All Class Probabilities")
    prob_dict = {class_names[i]: probabilities[0][i].item() for i in range(NUM_CLASSES)}
    
    # Sort by probability (highest first)
    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    
    for class_name, prob in sorted_probs:
        st.write(f"**{class_name}:** {prob:.2%}")
        st.progress(prob)
    
    # Additional information
    st.subheader("Model Information")
    st.write(f"**Confidence Threshold:** {CONFIDENCE_THRESHOLD:.0%}")
    st.write(f"**Model Architecture:** EfficientNet-B3")
    st.write(f"**Number of Classes:** {NUM_CLASSES}")