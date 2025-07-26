import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from src.dataset import CUBDataset
from src.model import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms (same as test)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load test dataset
test_dataset = CUBDataset('data', train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load model
model = build_model().to(device)
model.load_state_dict(torch.load("outputs/mobilenet_cub.pth"))
model.eval()

# Evaluate
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Accuracy
acc = accuracy_score(y_true, y_pred)
print(f"✅ Test Accuracy: {acc * 100:.2f}%")

# Classification report
print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred, digits=3))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, cmap='Blues', xticklabels=False, yticklabels=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("outputs/plots/confusion_matrix.png")
plt.show()
