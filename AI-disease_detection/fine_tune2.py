import torch
import timm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from torchvision.transforms import RandAugment

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class distribution from dataset
class_counts = {
    'bacterial_leaf_blight': 1981,
    'brown_spot': 1963,
    'healthy': 2008,
    'leaf_blast': 2003,
    'leaf_scald': 2190,
    'narrow_brown_spot': 1938,
    'neck_blast': 1000,  # Reduce weight for neck_blast
    'sheath_blight': 2120,
    'tungro': 2273
}

# Compute class weights (inverse frequency, reducing neck_blast weight)
total = sum(class_counts.values())
class_weights = {cls: total / (len(class_counts) * count) for cls, count in class_counts.items()}
class_weights["neck_blast"] *= 0.6  # Reduce weight since it's performing well

# Convert to tensor
weights = torch.tensor(list(class_weights.values()), dtype=torch.float32).to(device)

# Focal Loss Implementation
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=weights):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha)(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# Targeted Augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    RandAugment(),  # Stronger augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset & Dataloader
data_dir = "/home/ubuntu/yolo_disease/dataset_mix"
train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=train_transform)
test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "test"), transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)  # Reduce batch size
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

# Training Loop
def train_model(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = timm.create_model("efficientnet_b3", pretrained=True, num_classes=len(class_counts)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = FocalLoss()

    # Use mixed precision training for memory efficiency
    scaler = torch.cuda.amp.GradScaler()

    # New checkpoint save location
    checkpoint_dir = "/home/ubuntu/yolo_disease/fine_tune_op/FT2"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")

    start_epoch = 0
    best_acc = 0  # Initialize best accuracy

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print("✅ Model weights loaded successfully!")

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("✅ Optimizer state loaded successfully!")

        start_epoch = checkpoint.get("epoch", 0) + 1  # Default to 0 if not found
        best_acc = checkpoint.get("accuracy", 0)  # Default to 0 if not found

        print(f"🔄 Resuming training from epoch {start_epoch} (Seed {seed})")
    else:
        print(f"🚀 Starting new training run (Seed {seed})")

    num_epochs = 30
    accumulation_steps = 2  # Accumulate gradients over multiple steps

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():  # Use mixed precision
                outputs = model(images)
                loss = criterion(outputs, labels) / accumulation_steps  # Normalize loss

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or i == len(train_loader) - 1:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        avg_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        # Log results
        with open("log.txt", "a") as log_file:
            log_file.write(f"Seed {seed}, Epoch {epoch}, Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}\n")

        print(f"Seed {seed}, Epoch {epoch}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "accuracy": val_acc
            }, checkpoint_path)
            print(f"💾 New best model saved at epoch {epoch} with Val Acc: {val_acc:.4f}")

# Train with multiple seeds for an ensemble
for seed in [42, 1337, 2025]:
    train_model(seed)
