import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from timm import create_model
import numpy as np
import time
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from torch.amp import GradScaler, autocast

# =============== Device Config ===============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# =============== Transforms ===============
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =============== Load Dataset ===============
full_dataset = ImageFolder(root="dataset/train", transform=transform_train)
targets = full_dataset.targets

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
indices = list(range(len(full_dataset)))
train_idx, val_idx = next(sss.split(indices, targets))

train_dataset = Subset(full_dataset, train_idx)
valid_dataset = Subset(full_dataset, val_idx)
test_dataset = ImageFolder(root="dataset/test", transform=transform_test)

# =============== Class Weights ===============
class_counts = Counter(targets)
total_samples = sum(class_counts.values())
class_weights = [total_samples / class_counts[i] for i in range(len(class_counts))]
weights = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)

# =============== DataLoaders ===============
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)

# =============== Load Previous Model ===============
model = create_model("efficientnet_b3", pretrained=False, num_classes=len(full_dataset.classes))
model.load_state_dict(torch.load("sugarcane_disease_classifier.pth", map_location=device))
model = model.to(device)

# =============== Optimizer & AMP ===============
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
scaler = GradScaler()

# =============== Training Loop ===============
def train(model, train_loader, valid_loader, criterion, optimizer, total_epochs=20):
    model.train()
    start_time = time.time()

    for epoch in range(total_epochs):
        epoch_start = time.time()
        total_loss, total_val_loss = 0, 0
        correct, total, val_correct, val_total = 0, 0, 0, 0
        steps, val_steps = len(train_loader), len(valid_loader)
        all_preds, all_labels = [], []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast(device_type=device.type):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        model.eval()
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        model.train()

        epoch_loss = total_loss / steps
        epoch_acc = 100 * correct / total
        val_loss = total_val_loss / val_steps
        val_acc = 100 * val_correct / val_total
        epoch_time = time.time() - epoch_start
        elapsed_time = time.time() - start_time
        remaining_time = (elapsed_time / (epoch + 1)) * (total_epochs - epoch - 1)

        print(f"Epoch [{epoch+1}/{total_epochs}], Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Time: {epoch_time:.2f}s, "
              f"ETA: {remaining_time/60:.2f} min")

        if (epoch + 1) % 5 == 0:
            cm = confusion_matrix(all_labels, all_preds)
            class_names = full_dataset.classes
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=class_names, yticklabels=class_names)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(f"Validation Confusion Matrix (Epoch {epoch+1})")
            plt.show()

        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_loss,
                'train_accuracy': epoch_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
            }
            torch.save(checkpoint, f"checkpoint_epoch_{epoch+1}_finetune.pth")
            print(f"✅ Checkpoint saved at epoch {epoch+1}")

# =============== Test Function ===============
def test(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    print(f"🧪 Test Accuracy: {100 * correct / total:.2f}%")

# =============== Run Training ===============
train(model, train_loader, valid_loader, criterion, optimizer, total_epochs=20)
test(model, test_loader)
torch.save(model.state_dict(), "sugarcane_disease_classifier_finetuned.pth")
print("✅ Fine-tuned model saved as 'sugarcane_disease_classifier_finetuned.pth'")

# =============== Print Class Mapping ===============
print("📘 Class to Index Mapping:")
print(ImageFolder(root="dataset/train").class_to_idx)
