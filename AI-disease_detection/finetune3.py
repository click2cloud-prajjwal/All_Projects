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
    'neck_blast': 1000,
    'sheath_blight': 2120,
    'tungro': 2273
}

# Compute class weights
total = sum(class_counts.values())
class_weights = {cls: total / (len(class_counts) * count) for cls, count in class_counts.items()}
class_weights["neck_blast"] *= 0.6
class_weights["sheath_blight"] *= 0.8
class_weights["leaf_blast"] *= 1.2
class_weights["brown_spot"] *= 1.2

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
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(15),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    RandAugment(),
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

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

# Training Loop
def train_model(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = timm.create_model("efficientnet_b3", pretrained=True, num_classes=len(class_counts)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = FocalLoss()
    scaler = torch.amp.GradScaler()

    checkpoint_dir = "/home/ubuntu/yolo_disease/fine_tune_op/FT2"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"best_model_seed_{seed}.pth")  # Unique filename per seed

    start_epoch = 0
    best_acc = 0

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_acc = checkpoint.get("accuracy", 0)
        print(f"🔄 Resuming training from epoch {start_epoch} (Seed {seed})")
    else:
        print(f"🚀 Starting new training run (Seed {seed})")

    num_epochs = 30
    accumulation_steps = 2

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels) / accumulation_steps
            
            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or i == len(train_loader) - 1:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        scheduler.step()
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

for seed in [42, 1337, 2025]:
    train_model(seed)
