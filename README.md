# Vision Transformer vs. Swin Transformer for Prostate Cancer Detection

This repository contains code for prostate cancer detection using transformer-based deep learning architectures on the publicly available PI-CAI prostate cancer dataset.

The project compares:

- **Swin Transformer**
- **Vision Transformer (ViT)**
- **Hybrid Swin + ViT feature-fusion model**

## Manuscript Title

**Vision Transformer vs. Swin Transformer: Identifying the Optimal Architecture for Prostate Cancer Detection**

## Dataset

This project uses the Kaggle PI-CAI prostate cancer dataset:

```text
https://www.kaggle.com/datasets/varshithpsingh/prostate-cancer-pi-cai-dataset
```

In Kaggle notebooks, add the dataset using **Add Data** and keep the following path:

```text
/kaggle/input/prostate-cancer-pi-cai-dataset
```

The scripts assume the metadata file:

```text
/kaggle/input/prostate-cancer-pi-cai-dataset/Metadata with lesion info.csv
```

## Repository Structure

```text
prostate-transformer-detection/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── configs/
│   └── config.yaml
│
├── data/
│   └── README.md
│
├── notebooks/
│   ├── swin_transformer_original.ipynb
│   └── vit_transformer_original.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── train_utils.py
│   ├── train_swin.py
│   ├── train_vit.py
│   └── train_hybrid.py
│
├── models/
│   └── .gitkeep
│
└── outputs/
    └── .gitkeep
```

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/prostate-transformer-detection.git
cd prostate-transformer-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running on Kaggle

1. Create a new Kaggle notebook.
2. Add the PI-CAI dataset from Kaggle.
3. Upload or clone this repository.
4. Confirm the dataset path in `configs/config.yaml`.
5. Run the required script.

## Train Swin Transformer

```bash
python src/train_swin.py
```

## Train Vision Transformer

```bash
python src/train_vit.py
```

## Train Hybrid Swin + ViT Model

```bash
python src/train_hybrid.py
```

## Notes

The original notebooks are preserved in the `notebooks/` folder. The Python scripts in `src/` provide a cleaner GitHub-ready structure for reproducible experiments.

The dataset is not included in this repository due to size and licensing constraints. Please download it directly from Kaggle.

## Citation

If this repository supports your research, please cite the associated manuscript:

```bibtex
@article{prostate_transformer_detection,
  title={Vision Transformer vs. Swin Transformer: Identifying the Optimal Architecture for Prostate Cancer Detection},
  author={Your Name and Co-authors},
  journal={},
  year={2026}
}
```

## License

This project is released under the MIT License.
