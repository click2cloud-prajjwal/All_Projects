import os
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import AutoModelForImageClassification, AutoProcessor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from PIL import Image
from tqdm import tqdm
import timm  

# Load DinoV2 Model (pretrained on rice disease dataset)
dinov2_model = AutoModelForImageClassification.from_pretrained("cvmil/dinov2-base_rice-leaf-disease-augmented_fft")
dinov2_model.eval()
dinov2_processor = AutoProcessor.from_pretrained("cvmil/dinov2-base_rice-leaf-disease-augmented_fft")

# Load fine-tuned EfficientNet-B3 using timm for correct architecture
EB3_MODEL_PATH = "./checkpoint_epoch_30.pth"
checkpoint = torch.load(EB3_MODEL_PATH, map_location="cpu")

model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=9)
model.load_state_dict(checkpoint["model_state_dict"]) 
model.eval()
print("EfficientNet-B3 model loaded successfully!")

# Image preprocessing transformations
efficientnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_predictions(image):
    """Extract predictions from both DinoV2 and EfficientNetB3 models."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    # DinoV2 processing
    inputs = dinov2_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        dinov2_logits = dinov2_model(**inputs).logits.cpu().numpy()

    # EfficientNet processing
    image_tensor = efficientnet_transform(image).unsqueeze(0)
    with torch.no_grad():
        efficientnet_preds = model(image_tensor).cpu().numpy()

    # Stack predictions into one feature vector
    stacked_features = np.hstack([dinov2_logits, efficientnet_preds])
    return stacked_features

def load_dataset(root_folder):
    """Loads dataset from folder structure: root/class_x/*.jpg"""
    X, y = [], []
    class_map = {}

    for class_idx, class_name in enumerate(sorted(os.listdir(root_folder))):
        class_path = os.path.join(root_folder, class_name)
        if not os.path.isdir(class_path):
            continue
        
        class_map[class_name] = class_idx

        for img_file in tqdm(os.listdir(class_path), desc=f"Processing {class_name}"):
            img_path = os.path.join(class_path, img_file)
            try:
                image = Image.open(img_path)
                features = get_predictions(image)
                X.append(features)
                y.append(class_idx)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    return np.vstack(X), np.array(y), class_map

# Load train and test sets
train_folder = "./train"
test_folder = "./test"

X_train, y_train, class_map = load_dataset(train_folder)
X_test, y_test, _ = load_dataset(test_folder)

# Train Random Forest as meta-classifier
meta_model = RandomForestClassifier(n_estimators=100, random_state=42)
meta_model.fit(X_train, y_train)

# Save model and class_map using joblib
import joblib
joblib.dump(meta_model, "meta_model.pkl")
joblib.dump(class_map, "class_map.pkl")
print("Meta-classifier and class map saved successfully!")

# Evaluate on test set
y_pred = meta_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Stacked Model Accuracy: {accuracy:.2f}")

def stacked_predict(image_path):
    """Predict disease class using the stacked model."""
    image = Image.open(image_path)
    features = get_predictions(image)
    
    # Load model and class map
    meta_model = joblib.load("meta_model.pkl")
    class_map = joblib.load("class_map.pkl")
    
    prediction = meta_model.predict(features)
    class_label = [key for key, val in class_map.items() if val == prediction][0]
    return class_label

# Example usage
test_image_path = "./bacterial-blight-of-rice-rice-1581498605.jpg"
predicted_label = stacked_predict(test_image_path)
print("Predicted Class:", predicted_label)
import joblib
class_map_script = joblib.load("./class_map.pkl")
print("Class Map in Script:", class_map_script)
