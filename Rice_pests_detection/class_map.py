import os
import joblib
from torchvision.datasets import ImageFolder
from torchvision import transforms

# === Path to the training folder ===
train_folder = "/home/ubuntu/pests_and_disease/Rice_pest_detection/dataset/train"

# === Define a simple transform just to load the dataset ===
dummy_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# === Load dataset using ImageFolder ===
dataset = ImageFolder(root=train_folder, transform=dummy_transform)

# === Get the class-to-index mapping ===
class_map = dataset.class_to_idx  # e.g., {'Grassy shoot': 0, ..., 'Unknown': 8}

# === Save the mapping to a file ===
output_path = "Rice_pest_class_map.pkl"
joblib.dump(class_map, output_path)

# === Confirmation log ===
print("✅ Rice_pest_class_map.pkl has been saved at:", output_path)
print("📦 Class Mapping:")
for cls, idx in class_map.items():
    print(f"{idx:2} → {cls}")