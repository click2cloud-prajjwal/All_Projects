import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from timm import create_model
import time
import numpy as np

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# Data Augmentation & Preprocessing
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),  # Ensure input is converted to Tensor before AutoAugment
    transforms.RandomErasing(p=0.2),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load Dataset
full_dataset = ImageFolder(root="/home/ubuntu/yolo_disease/dataset_mix/train", transform=transform_train)
train_size = int(0.8 * len(full_dataset))
valid_size = len(full_dataset) - train_size
train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])

test_dataset = ImageFolder(root="/home/ubuntu/yolo_disease/dataset_mix/test", transform=transform_test)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)

# Load EfficientNet-B3 Model (Pretrained)
model = create_model("efficientnet_b0", pretrained=True, num_classes=len(full_dataset.classes))
model = model.to(device)

# Loss Function & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

# Training Loop with Validation & Checkpoint Saving
from torch.amp import GradScaler, autocast
scaler = GradScaler("cuda")

def train(model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs=30):
    model.train()
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        total_loss, total_val_loss = 0, 0
        correct, total, val_correct, val_total = 0, 0, 0, 0
        steps, val_steps = len(train_loader), len(valid_loader)

        # Training Loop
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with autocast("cuda"):  # Enable mixed precision
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient Clipping
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        scheduler.step(total_loss / steps)

        # Validation Loop
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

        model.train()

        # Compute Loss & Accuracy
        epoch_loss = total_loss / steps
        epoch_acc = 100 * correct / total
        val_loss = total_val_loss / val_steps
        val_acc = 100 * val_correct / val_total

        epoch_time = time.time() - epoch_start
        elapsed_time = time.time() - start_time
        remaining_time = (elapsed_time / (epoch + 1)) * (num_epochs - epoch - 1)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Time: {epoch_time:.2f}s, "
              f"ETA: {remaining_time/60:.2f} min")

        # Save checkpoint every 5 epochs
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
            torch.save(checkpoint, f"checkpoint_epoch_{epoch+1}.pth")
            print(f"Checkpoint saved at epoch {epoch+1}")

# Test Function
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
    print(f"Test Accuracy: {100 * correct/total:.2f}%")

# Train & Evaluate
train(model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs=30)
test(model, test_loader)

# Save Final Model
torch.save(model.state_dict(), "paddy_disease_classifier.pth")
print("Final model saved as 'paddy_disease_classifier.pth'")

# Print Class-to-Index Mapping
dataset = ImageFolder(root="/home/ubuntu/yolo_disease/dataset_mix/train")
print(dataset.class_to_idx)