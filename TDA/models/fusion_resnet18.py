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

        # TDA MLP branch
        hidden_dims = [1024, 512, 256, 128]
        layers = []
        prev_dim = tda_input_dim
        for i, hdim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.BatchNorm1d(hdim))
            layers.append(nn.ReLU())
            if i >= 2:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hdim
        self.tda_mlp = nn.Sequential(*layers)
        self.tda_fc_dim = hidden_dims[-1]

        # Fusion MLP head
        self.classifier = nn.Sequential(
            nn.Linear(self.cnn_fc_dim + self.tda_fc_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, image, tda):
        # image: (B, 3, H, W), tda: (B, input_dim)
        cnn_feat = self.cnn_backbone(image).view(image.size(0), -1)  # (B, 512)
        tda_feat = self.tda_mlp(tda)                                # (B, 128)
        fused = torch.cat([cnn_feat, tda_feat], dim=1)              # (B, 640)
        return self.classifier(fused)                               # (B, 17)