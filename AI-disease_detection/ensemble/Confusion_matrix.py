import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
from wrapper import EnsembleModel, model_paths  # Import your ensemble model wrapper

# Load device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the saved ensemble model
ensemble_model = EnsembleModel(model_paths)
ensemble_model.load_state_dict(torch.load("ensemble_model.pth", map_location=device))
ensemble_model.to(device)
ensemble_model.eval()

# Define transformations (same as used during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ensure consistency with training
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load test dataset
test_dataset = ImageFolder("/home/ubuntu/yolo_disease/mix_dataset/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Get class names
class_names = test_dataset.classes

# Store predictions and true labels
all_preds = []
all_labels = []

# Run inference with ensemble model
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = ensemble_model(images)  # Get predictions from the ensemble
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Ensemble Model")
plt.savefig("confusion_matrix-Ensemble_Model.png")

# Print classification report
print(classification_report(all_labels, all_preds, target_names=class_names))
