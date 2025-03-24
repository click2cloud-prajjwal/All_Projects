import torch
import torchvision.transforms as transforms
from PIL import Image
from wrapper import EnsembleModel, model_paths

# Load the saved ensemble model
ensemble_model = EnsembleModel(model_paths)
ensemble_model.load_state_dict(torch.load("ensemble_model.pth", map_location="cpu"))
ensemble_model.eval()

# Preprocess function
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# Load and predict
image_path = "GradCAM/rice_test_img/RiceSheathArk.jpg"
image = preprocess_image(image_path)

with torch.no_grad():
    probs = ensemble_model(image)
    predicted_class = torch.argmax(probs, dim=1).item()

class_labels = {
    0: "Bacterial Leaf Blight",
    1: "Brown Spot",
    2: "Healthy",
    3: "Leaf Blast",
    4: "Leaf Scald",
    5: "Narrow Brown Spot",
    6: "Sheath Blight",
    7: "Tungro"
}

print(f"Predicted Class: {predicted_class} - {class_labels.get(predicted_class, 'Unknown')}")
