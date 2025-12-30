import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import random
import numpy as np
import os
import csv

# -----------------------------
# SEED
# -----------------------------
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# -----------------------------
# AYARLAR
# -----------------------------
TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"
MODEL_PATH = "trained_models/model.pth"
CLASSES_PATH = "trained_models/classes.json"
LOG_PATH = "trained_models/training_log.csv"
METRICS_DIR = "metrics"
BATCH_SIZE = 32
EPOCHS = 15
LR = 3e-4
IMG_SIZE = 224

os.makedirs("trained_models", exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Cihaz:", device)

# -----------------------------
# TRANSFORMLAR
# -----------------------------
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# -----------------------------
# DATASET & DATALOADER
# -----------------------------
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

class_names = train_dataset.classes
print("Sınıflar:", class_names)

with open(CLASSES_PATH, "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False)

# -----------------------------
# MODEL
# -----------------------------
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# LOG DOSYASI
# -----------------------------
with open(LOG_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Loss", 
                     "Train_Acc", "Train_Precision", "Train_Recall",
                     "Test_Acc", "Test_Precision", "Test_Recall"])

# -----------------------------
# EĞİTİM
# -----------------------------
train_losses, train_accuracies, train_precisions, train_recalls = [], [], [], []
test_accuracies, test_precisions, test_recalls = [], [], []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    y_true_train, y_pred_train = [], []

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        y_true_train.extend(labels.cpu().numpy())
        y_pred_train.extend(predicted.cpu().numpy())

    train_accuracy = accuracy_score(y_true_train, y_pred_train)
    train_precision = precision_score(y_true_train, y_pred_train, average="macro")
    train_recall = recall_score(y_true_train, y_pred_train, average="macro")

    # -----------------------------
    # TEST
    # -----------------------------
    model.eval()
    y_true_test, y_pred_test = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true_test.extend(labels.cpu().numpy())
            y_pred_test.extend(predicted.cpu().numpy())

    test_accuracy = accuracy_score(y_true_test, y_pred_test)
    test_precision = precision_score(y_true_test, y_pred_test, average="macro")
    test_recall = recall_score(y_true_test, y_pred_test, average="macro")

    # -----------------------------
    # LOG & PRINT
    # -----------------------------
    print(f"Epoch {epoch+1}/{EPOCHS} - "
          f"Loss: {running_loss / len(train_loader):.4f} - "
          f"Train Acc: {train_accuracy*100:.2f}%, Precision: {train_precision*100:.2f}%, Recall: {train_recall*100:.2f}% - "
          f"Test Acc: {test_accuracy*100:.2f}%, Precision: {test_precision*100:.2f}%, Recall: {test_recall*100:.2f}%")

    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, running_loss/len(train_loader),
                         train_accuracy, train_precision, train_recall,
                         test_accuracy, test_precision, test_recall])

    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(train_accuracy)
    train_precisions.append(train_precision)
    train_recalls.append(train_recall)

    test_accuracies.append(test_accuracy)
    test_precisions.append(test_precision)
    test_recalls.append(test_recall)

# -----------------------------
# CONFUSION MATRIX & METRİKLER
# -----------------------------
cm = confusion_matrix(y_true_test, y_pred_test)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(METRICS_DIR, "confusion_matrix.png"))
plt.close()

epochs_range = range(1, EPOCHS+1)
plt.figure(); plt.plot(epochs_range, train_losses); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Train Loss"); plt.savefig(os.path.join(METRICS_DIR,"train_loss.png")); plt.close()
plt.figure(); plt.plot(epochs_range, train_accuracies, label="Train"); plt.plot(epochs_range, test_accuracies, label="Test"); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy"); plt.legend(); plt.savefig(os.path.join(METRICS_DIR,"accuracy.png")); plt.close()
plt.figure(); plt.plot(epochs_range, train_precisions, label="Train"); plt.plot(epochs_range, test_precisions, label="Test"); plt.xlabel("Epoch"); plt.ylabel("Precision"); plt.title("Precision"); plt.legend(); plt.savefig(os.path.join(METRICS_DIR,"precision.png")); plt.close()
plt.figure(); plt.plot(epochs_range, train_recalls, label="Train"); plt.plot(epochs_range, test_recalls, label="Test"); plt.xlabel("Epoch"); plt.ylabel("Recall"); plt.title("Recall"); plt.legend(); plt.savefig(os.path.join(METRICS_DIR,"recall.png")); plt.close()

# -----------------------------
# MODEL KAYDET
# -----------------------------
torch.save(model.state_dict(), MODEL_PATH)
print("Model kaydedildi:", MODEL_PATH)
