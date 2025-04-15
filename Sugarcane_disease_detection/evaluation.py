import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from transformers import SwinForImageClassification, AutoImageProcessor
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# Constants
VAL_SPLIT = 0.2  # 20% for validation
BATCH_SIZE = 8
CHECKPOINT_PATH = "model_chkpt/swin_tiny_epoch10.pth"  # Path to your checkpoint
NUM_CLASSES = 10  # Adjust this based on your dataset's number of classes
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
DATA_DIR = "balanced_dataset"  # Replace with your dataset path

# Load the image processor from the base model
processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

# Dataset & Dataloader
transform = transforms.Compose([
    transforms.Resize((processor.size['height'], processor.size['width'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
])

# Load the full dataset
full_ds = datasets.ImageFolder(root=DATA_DIR, transform=transform)

# Split the dataset into training and validation
train_size = int((1 - VAL_SPLIT) * len(full_ds))  # 80% for training
val_size = len(full_ds) - train_size  # 20% for validation
train_ds, val_ds = random_split(full_ds, [train_size, val_size])

# Dataloaders
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Load the model and replace classification head
model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
model.to(DEVICE)

# Load the checkpoint
checkpoint = torch.load(CHECKPOINT_PATH)
model.load_state_dict(checkpoint['model_state_dict'])  # Load model weights from checkpoint
model.eval()  # Set the model to evaluation mode

# Inference & evaluation
total_loss, total_correct, total = 0, 0, 0
criterion = torch.nn.CrossEntropyLoss()
all_preds, all_labels = [], []

for batch in tqdm(val_loader, desc="Evaluating"):
    inputs, labels = batch
    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

    with torch.no_grad():
        outputs = model(inputs)
        loss = criterion(outputs.logits, labels)
        total_loss += loss.item() * labels.size(0)

        preds = torch.argmax(outputs.logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Final metrics
val_loss = total_loss / total
val_accuracy = total_correct / total * 100
print(f"\n✅ Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

# Debugging 🔍

# 1. Print few predictions vs ground truths
print("\n🎯 Sample Predictions vs Ground Truths:")
for i in range(10):
    print(f"Pred: {all_preds[i]}, True: {all_labels[i]}")

# 2. Classification report
print("\n📋 Classification Report:")
print(classification_report(all_labels, all_preds))

# 3. Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(xticks_rotation=45)
plt.title("🧾 Confusion Matrix")
plt.tight_layout()
plt.show()
