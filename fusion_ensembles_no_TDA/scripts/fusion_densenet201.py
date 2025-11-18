import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet201, DenseNet201_Weights


class TDAFusionDenseNet201(nn.Module):
    """
    DenseNet201 backbone fused with a fully-connected branch for TDA features.
    """

    def __init__(self, tda_input_dim=18000, num_classes=17, dropout_rate=0.5):
        super().__init__()

        backbone = densenet201(weights=DenseNet201_Weights.DEFAULT)
        self.features = backbone.features
        self.num_features = backbone.classifier.in_features

        self.cnn_dropout = nn.Dropout(p=dropout_rate * 0.5)

        hidden_dims = [1024, 512, 256, 128]
        mlp_layers = []
        prev_dim = tda_input_dim
        for i, hdim in enumerate(hidden_dims):
            mlp_layers.append(nn.Linear(prev_dim, hdim))
            mlp_layers.append(nn.BatchNorm1d(hdim))
            mlp_layers.append(nn.ReLU(inplace=True))
            drop_p = dropout_rate if i >= 1 else dropout_rate * 0.5
            mlp_layers.append(nn.Dropout(drop_p))
            prev_dim = hdim
        self.tda_mlp = nn.Sequential(*mlp_layers)
        self.tda_fc_dim = hidden_dims[-1]

        fusion_dim = self.num_features + self.tda_fc_dim
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.fusion_dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, image: torch.Tensor, tda: torch.Tensor) -> torch.Tensor:
        # image: (B, 3, H, W), tda: (B, tda_dim)
        features = self.features(image)
        features = F.relu(features, inplace=True)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        cnn_feat = torch.flatten(features, 1)
        cnn_feat = self.cnn_dropout(cnn_feat)

        tda_feat = self.tda_mlp(tda)

        fused = torch.cat([cnn_feat, tda_feat], dim=1)
        fused = self.fusion_norm(fused)
        fused = self.fusion_dropout(fused)
        return self.classifier(fused)
