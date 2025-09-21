 import os
import torch
import torchvision
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# === Step 1: Setup GPU/CPU ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)

# === Step 2: Image Preprocessing (for VGG16) ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet mean
                         [0.229, 0.224, 0.225])   # ImageNet std
])

# === Step 3: Load Dataset ===
data_dir = '/kaggle/input/cellnet-beta-version/CellNet/CellNet'
full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# === Step 4: Train/Test Split ===
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

class_names = full_dataset.classes
num_classes = len(class_names)
print(f"🧪 Classes: {class_names}")

# === Step 5: Load VGG16 Pretrained Model ===
model = models.vgg16(pretrained=True)

# Freeze convolutional layers
for param in model.features.parameters():
    param.requires_grad = False

# Replace classifier for 20 classes
model.classifier[6] = nn.Linear(4096, num_classes)
model = model.to(device)

# === Step 6: Loss & Optimizer ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.0001)

# === Step 7: Training Loop ===
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    acc = 100 * correct / total
    print(f"✅ Epoch [{epoch+1}/{epochs}] - Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%")

# === Step 7.5: Save Trained Model ===
model_save_path = 'vgg16_cellnet.pth'
torch.save(model.state_dict(), model_save_path)
print(f"📁 Model saved to: {model_save_path}")

# === Step 8: Evaluation ===
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# === Step 9: Evaluation Metrics ===
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

print("\n📊 Evaluation Metrics:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# === Step 10: Full Classification Report ===
print("\n🧾 Detailed Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

# === Step 11: Confusion Matrix Plot ===
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
