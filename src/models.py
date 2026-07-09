import torch
import torch.nn as nn
import timm


def create_swin_model(model_name="swin_base_patch4_window7_224", num_classes=2, pretrained=True):
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)


def create_vit_model(model_name="vit_base_patch16_224", num_classes=2, pretrained=True):
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)


def create_feature_extractor(model_name, pretrained=True):
    return timm.create_model(model_name, pretrained=pretrained, num_classes=0)


class HybridMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_classes=2, dropout=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)
