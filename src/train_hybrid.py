from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from config import load_config
from dataset import ProstateImageDataset
from preprocessing import build_processed_image_dataset
from models import create_feature_extractor, HybridMLP
from train_utils import set_seed, plot_confusion_matrix, print_report


def extract_features(model, dataloader, device):
    model.eval()
    features, labels = [], []

    with torch.no_grad():
        for images, target in dataloader:
            images = images.to(device)

            if hasattr(model, "forward_features"):
                feat = model.forward_features(images)
            else:
                feat = model(images)

            if feat.ndim == 4:
                feat = feat.mean(dim=(-2, -1))
            elif feat.ndim == 3:
                feat = feat.mean(dim=1)

            features.append(feat.cpu().numpy())
            labels.append(target.numpy())

    return np.vstack(features), np.concatenate(labels)


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

    transform = transforms.Compose([
        transforms.Resize((cfg["preprocessing"]["image_size"], cfg["preprocessing"]["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_loader = DataLoader(ProstateImageDataset(train_df, transform), batch_size=cfg["training"]["batch_size"], shuffle=False)
    val_loader = DataLoader(ProstateImageDataset(val_df, transform), batch_size=cfg["training"]["batch_size"], shuffle=False)
    test_loader = DataLoader(ProstateImageDataset(test_df, transform), batch_size=cfg["training"]["batch_size"], shuffle=False)

    swin_extractor = create_feature_extractor(cfg["models"]["swin_name"], pretrained=True).to(device)
    vit_extractor = create_feature_extractor(cfg["models"]["vit_name"], pretrained=True).to(device)

    train_swin, train_y = extract_features(swin_extractor, train_loader, device)
    val_swin, val_y = extract_features(swin_extractor, val_loader, device)
    test_swin, test_y = extract_features(swin_extractor, test_loader, device)

    train_vit, _ = extract_features(vit_extractor, train_loader, device)
    val_vit, _ = extract_features(vit_extractor, val_loader, device)
    test_vit, _ = extract_features(vit_extractor, test_loader, device)

    train_features = np.concatenate([train_swin, train_vit], axis=1)
    val_features = np.concatenate([val_swin, val_vit], axis=1)
    test_features = np.concatenate([test_swin, test_vit], axis=1)

    train_ds = TensorDataset(torch.tensor(train_features, dtype=torch.float32), torch.tensor(train_y, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(val_features, dtype=torch.float32), torch.tensor(val_y, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(test_features, dtype=torch.float32), torch.tensor(test_y, dtype=torch.long))

    train_loader_f = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader_f = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader_f = DataLoader(test_ds, batch_size=32, shuffle=False)

    model = HybridMLP(train_features.shape[1], num_classes=cfg["models"]["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    best_val_acc = 0.0
    Path(cfg["outputs"]["model_dir"]).mkdir(exist_ok=True)
    best_path = Path(cfg["outputs"]["model_dir"]) / "hybrid_swin_vit_mlp.pth"

    for epoch in range(20):
        model.train()
        for x, y in train_loader_f:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for x, y in val_loader_f:
                x, y = x.to(device), y.to(device)
                output = model(x)
                preds.extend(output.argmax(1).cpu().numpy())
                targets.extend(y.cpu().numpy())

        val_acc = accuracy_score(targets, preds)
        print(f"Hybrid Epoch {epoch + 1}/20 | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for x, y in test_loader_f:
            x, y = x.to(device), y.to(device)
            output = model(x)
            preds.extend(output.argmax(1).cpu().numpy())
            targets.extend(y.cpu().numpy())

    print("Hybrid Test Accuracy:", accuracy_score(targets, preds))
    print_report(targets, preds)

    Path(cfg["outputs"]["result_dir"]).mkdir(exist_ok=True)
    plot_confusion_matrix(targets, preds, title="Hybrid Swin + ViT Confusion Matrix",
                          save_path=Path(cfg["outputs"]["result_dir"]) / "hybrid_confusion_matrix.png")


if __name__ == "__main__":
    main()
