import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

from src.dataset import CUBDataset
from src.model import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------- Transforms ---------------------
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --------------------- Datasets & Dataloaders ---------------------
train_dataset = CUBDataset('data', train=True, transform=transform_train)
test_dataset = CUBDataset('data', train=False, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

# --------------------- Model ---------------------
model = build_model().to(device)

# --------------------- Loss, Optimizer, Scheduler ---------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# --------------------- Training Loop ---------------------
def train(num_epochs=30):
    best_acc = 0.0
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        total_train_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = total_train_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # --- Evaluation ---
        model.eval()
        total_test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                loss = criterion(outputs, labels)
                total_test_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_loss = total_test_loss / len(test_loader)
        test_acc = correct / total
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

        # Save best model
        if test_acc > best_acc:
          best_acc = test_acc
          print(f"New best model with accuracy {best_acc:.4f}, saving model...")
          torch.save(model.state_dict(), "outputs/mobilenet_cub.pth")


        scheduler.step()

    return train_losses, test_losses, train_accuracies, test_accuracies


# --------------------- Plotting ---------------------
def plot_curves(train_losses, test_losses, train_acc, test_acc):
    os.makedirs("outputs/plots", exist_ok=True)

    # Loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.savefig("outputs/plots/loss_curve.png")

    # Accuracy curve
    plt.figure(figsize=(8, 5))
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(test_acc, label="Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.savefig("outputs/plots/accuracy_curve.png")

# --------------------- Main ---------------------
if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    train_losses, test_losses, train_acc, test_acc = train(num_epochs=30)
    plot_curves(train_losses, test_losses, train_acc, test_acc)
