from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class ProstateImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        label = torch.tensor(int(row["label"]), dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label


def normalize_binary_label(value):
    text = str(value).strip().lower()
    positive_values = {"1", "yes", "positive", "cancer", "malignant", "clinically significant", "csPCa"}
    try:
        return 1 if float(text) > 0 else 0
    except Exception:
        return 1 if text in positive_values else 0


def load_metadata(metadata_file):
    df = pd.read_csv(metadata_file)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    return df
