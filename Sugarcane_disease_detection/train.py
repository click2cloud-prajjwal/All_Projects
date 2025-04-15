import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from transformers import AutoImageProcessor, SwinForImageClassification
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import os
from tqdm import tqdm

# Configs
MODEL_ID = "rmezapi/sugarcane-diagnosis-swin-tiny"
BASE_MODEL_ID = "microsoft/swin-tiny-patch4-window7-224"  # Base model for image processor
DATA_DIR = "balanced_dataset"
NUM_CLASSES = 10
BATCH_SIZE = 32
EPOCHS = 15
WARMUP_EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_SPLIT = 0.2  # 20% for validation

# Load the image processor from the base model
processor = AutoImageProcessor.from_pretrained(BASE_MODEL_ID)

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
model = SwinForImageClassification.from_pretrained(MODEL_ID)
model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
model.to(DEVICE)

# Freeze encoder for warmup
def freeze_encoder(model):
    for name, param in model.named_parameters():
        if not name.startswith("classifier"):
            param.requires_grad = False

def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True

freeze_encoder(model)

# Optimizer & scheduler
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# Mixed precision
scaler = torch.amp.GradScaler()

# Training and validation loop
criterion = nn.CrossEntropyLoss()

# Helper function for validation
def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with torch.amp.autocast(DEVICE):
                outputs = model(images).logits
                loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    val_accuracy = (correct / total) * 100
    val_loss = val_loss / len(val_loader)
    
    return val_loss, val_accuracy

# Training loop
for epoch in range(EPOCHS):
    model.train()
    if epoch == WARMUP_EPOCHS:
        print("🧠 Unfreezing encoder...")
        unfreeze_all(model)
        optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images).logits
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "Acc": f"{(correct/total)*100:.2f}%"
        })

    scheduler.step()

    # Validation step
    val_loss, val_acc = validate(model, val_loader, criterion)
    print(f"Epoch {epoch+1} Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

    # Save checkpoint
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss,
        'accuracy': (correct / total) * 100
    }
    torch.save(checkpoint, f"swin_tiny_epoch{epoch+1}.pth")

print("✅ Training complete.")
# Save the final model
model.save_pretrained(MODEL_ID)
processor.save_pretrained(MODEL_ID)
print(f"Model saved to {MODEL_ID}")
# Clean up
torch.cuda.empty_cache()
print("🧹 Cache cleared.")