import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class TDAFusionResNet18(nn.Module):
    def __init__(self, tda_input_dim=18000, num_classes=17, dropout_rate=0.5):
        super().__init__()

        # CNN Backbone: ResNet18 without final FC
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-1])  # remove FC
        self.cnn_fc_dim = 512  # ResNet18 output from avgpool
        self.cnn_dropout = nn.Dropout(p=dropout_rate * 0.5)

        # TDA MLP branch
        hidden_dims = [1024, 512, 256, 128]
        layers = []
        prev_dim = tda_input_dim
        for i, hdim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.BatchNorm1d(hdim))
            layers.append(nn.ReLU())
            drop_p = dropout_rate if i >= 1 else dropout_rate * 0.5
            layers.append(nn.Dropout(drop_p))
            prev_dim = hdim
        self.tda_mlp = nn.Sequential(*layers)
        self.tda_fc_dim = hidden_dims[-1]

        # Fusion MLP head
        self.classifier = nn.Sequential(
            nn.Linear(self.cnn_fc_dim + self.tda_fc_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        self.fusion_norm = nn.LayerNorm(self.cnn_fc_dim + self.tda_fc_dim)
        self.fusion_dropout = nn.Dropout(dropout_rate)

    def forward(self, image, tda):
        # image: (B, 3, H, W), tda: (B, input_dim)
        cnn_feat = self.cnn_backbone(image).view(image.size(0), -1)  # (B, 512)
        cnn_feat = self.cnn_dropout(cnn_feat)
        tda_feat = self.tda_mlp(tda)                                # (B, 128)
        fused = torch.cat([cnn_feat, tda_feat], dim=1)              # (B, 640)
        fused = self.fusion_norm(fused)
        fused = self.fusion_dropout(fused)
        return self.classifier(fused)                               # (B, 17)
