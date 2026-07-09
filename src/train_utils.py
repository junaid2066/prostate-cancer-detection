import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler=None):
    model.train()
    losses, predictions, targets = [], [], []

    for images, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        predictions.extend(outputs.argmax(dim=1).detach().cpu().numpy())
        targets.extend(labels.detach().cpu().numpy())

    return float(np.mean(losses)), accuracy_score(targets, predictions)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    losses, predictions, targets = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluation"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            losses.append(loss.item())
            predictions.extend(outputs.argmax(dim=1).detach().cpu().numpy())
            targets.extend(labels.detach().cpu().numpy())

    return float(np.mean(losses)), accuracy_score(targets, predictions), np.array(targets), np.array(predictions)


def print_report(y_true, y_pred, class_names=("Benign", "Malignant")):
    print(classification_report(y_true, y_pred, target_names=list(class_names)))


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", save_path=None):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_training_curves(history, title, save_path=None):
    plt.figure(figsize=(7, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title(f"{title} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    if save_path:
        plt.savefig(save_path.replace(".png", "_loss.png"), dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Validation Accuracy")
    plt.title(f"{title} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    if save_path:
        plt.savefig(save_path.replace(".png", "_accuracy.png"), dpi=300, bbox_inches="tight")
    plt.show()
