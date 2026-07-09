import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

from config import load_config
from dataset import ProstateImageDataset
from preprocessing import build_processed_image_dataset
from models import create_swin_model
from train_utils import set_seed, train_one_epoch, evaluate, print_report, plot_confusion_matrix, plot_training_curves


def main():
    cfg = load_config()
    set_seed(cfg["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    processed_csv = Path(cfg["dataset"]["output_dir"]) / "processed_labels.csv"

    if processed_csv.exists():
        df = pd.read_csv(processed_csv)
    else:
        df = build_processed_image_dataset(
            root_dir=cfg["dataset"]["root_dir"],
            metadata_file=cfg["dataset"]["metadata_file"],
            output_dir=cfg["dataset"]["output_dir"],
            image_size=cfg["preprocessing"]["image_size"],
            num_slices=cfg["preprocessing"]["num_slices"],
        )

    train_df, test_df = train_test_split(
        df,
        test_size=cfg["training"]["test_size"],
        stratify=df["label"],
        random_state=cfg["training"]["seed"],
    )
    train_df, val_df = train_test_split(
        train_df,
        test_size=cfg["training"]["validation_size"],
        stratify=train_df["label"],
        random_state=cfg["training"]["seed"],
    )

    train_transform = transforms.Compose([
        transforms.Resize((cfg["preprocessing"]["image_size"], cfg["preprocessing"]["image_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((cfg["preprocessing"]["image_size"], cfg["preprocessing"]["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_loader = DataLoader(
        ProstateImageDataset(train_df, train_transform),
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
    )
    val_loader = DataLoader(
        ProstateImageDataset(val_df, val_transform),
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
    )
    test_loader = DataLoader(
        ProstateImageDataset(test_df, val_transform),
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
    )

    model = create_swin_model(cfg["models"]["swin_name"], cfg["models"]["num_classes"], pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(cfg["training"]["epochs"]):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch + 1}/{cfg['training']['epochs']} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    print("Test Accuracy:", test_acc)
    print_report(y_true, y_pred)

    Path(cfg["outputs"]["model_dir"]).mkdir(exist_ok=True)
    Path(cfg["outputs"]["result_dir"]).mkdir(exist_ok=True)

    torch.save(model.state_dict(), Path(cfg["outputs"]["model_dir"]) / "swin_transformer.pth")
    plot_confusion_matrix(y_true, y_pred, title="Swin Transformer Confusion Matrix",
                          save_path=Path(cfg["outputs"]["result_dir"]) / "swin_confusion_matrix.png")
    plot_training_curves(history, title="Swin Transformer",
                         save_path=str(Path(cfg["outputs"]["result_dir"]) / "swin_curves.png"))


if __name__ == "__main__":
    main()
