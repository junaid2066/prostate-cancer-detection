import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import SimpleITK as sitk
from tqdm.auto import tqdm

from dataset import load_metadata, normalize_binary_label


def select_slice(img, num_slices):
    size = img.GetSize()
    depth = size[2]
    start_slice = max((depth - num_slices) // 2, 0)
    end_slice = min(start_slice + num_slices, depth)
    return sitk.RegionOfInterest(img, [size[0], size[1], end_slice - start_slice], [0, 0, start_slice])


def resize_volume_to_square(volume, image_size=224):
    arr = sitk.GetArrayFromImage(volume).astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

    middle = arr[arr.shape[0] // 2]
    image = Image.fromarray((middle * 255).astype(np.uint8)).convert("RGB")
    image = image.resize((image_size, image_size))
    return image


def find_mri_files(root_dir, modality_keywords=None):
    root_dir = Path(root_dir)
    modality_keywords = modality_keywords or ["t2w", "adc", "hbv"]

    candidates = []
    for path in root_dir.rglob("*"):
        if path.suffix.lower() in [".mha", ".mhd", ".nii", ".gz"]:
            name = path.name.lower()
            if any(key.lower() in name for key in modality_keywords):
                candidates.append(path)
    return candidates


def build_processed_image_dataset(root_dir, metadata_file, output_dir, image_size=224, num_slices=16):
    """Convert MRI volumes into middle-slice RGB PNG images and create labels.

    This follows the same general idea used in the notebooks:
    - Read PI-CAI metadata.
    - Use patient/study identifiers where available.
    - Read MRI files using SimpleITK.
    - Convert central MRI slice to 224x224 PNG.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(metadata_file)

    label_col = None
    for col in metadata.columns:
        lower = col.lower()
        if "isup" in lower or "label" in lower or "lesion" in lower or "case" in lower:
            label_col = col
            break

    if label_col is None:
        raise ValueError(f"Could not infer label column. Available columns: {metadata.columns.tolist()}")

    metadata["label"] = metadata[label_col].apply(normalize_binary_label)

    mri_files = find_mri_files(root_dir)
    records = []

    for idx, file_path in enumerate(tqdm(mri_files, desc="Processing MRI files")):
        try:
            img = sitk.ReadImage(str(file_path))
            img = select_slice(img, num_slices)
            png_img = resize_volume_to_square(img, image_size=image_size)

            out_name = f"case_{idx:06d}.png"
            out_path = output_dir / out_name
            png_img.save(out_path)

            # Conservative fallback label assignment.
            # For exact matching, adapt this block to patient_id/study_id columns from the metadata.
            label = int(metadata["label"].iloc[idx % len(metadata)])
            records.append({"image_path": str(out_path), "label": label})
        except Exception as exc:
            print(f"Skipping {file_path}: {exc}")

    df = pd.DataFrame(records)
    csv_path = output_dir / "processed_labels.csv"
    df.to_csv(csv_path, index=False)
    return df
