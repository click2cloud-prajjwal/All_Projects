import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
from transformers import SwinForImageClassification, AutoImageProcessor
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import KFold

# Constants
VAL_SPLIT = 0.2  # 20% for validation
BATCH_SIZE = 8
CHECKPOINT_PATH = "model_chkpt/swin_tiny_epoch10.pth"  # Path to your checkpoint
NUM_CLASSES = 10  # Adjust this based on your dataset's number of classes
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
DATA_DIR = "balanced_dataset"  # Replace with your dataset path
K_FOLDS = 5  # Number of folds for cross-validation

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

# Cross-validation setup
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

# Start cross-validation
for fold, (train_idx, val_idx) in enumerate(kf.split(full_ds)):
    print(f"Fold {fold+1}/{K_FOLDS}")
    
    # Split dataset into train and validation for this fold
    train_ds = torch.utils.data.Subset(full_ds, train_idx)
    val_ds = torch.utils.data.Subset(full_ds, val_idx)

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Load the model and replace classification head
    model = SwinForImageClassification.from_pretrained("rmezapi/sugarcane-diagnosis-swin-tiny")
    model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    model.to(DEVICE)

    # Optionally load checkpoint
    if CHECKPOINT_PATH:
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])

    # Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # Training loop (Optional for each fold, but for full validation you'd typically load pre-trained models)
    model.train()
    for epoch in range(1):  # You can increase epochs here
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1} Training Loss: {running_loss/len(train_loader)}")

    # Evaluation after training
    model.eval()
    total_loss, total_correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            inputs, labels = batch
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(outputs.logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Final metrics for this fold
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
    plt.title(f"🧾 Confusion Matrix (Fold {fold+1})")
    plt.tight_layout()
    plt.show()

