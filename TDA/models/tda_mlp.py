import torch.nn as nn

class TDAOnlyMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout_rate: float = 0.5):
        super().__init__()

        # Hidden dimensions — can be tuned, but let's use decreasing pattern
        hidden_dims = [1024, 512, 256, 128]

        # 4-layer MLP for TDA features
        layers = []
        prev_dim = input_dim
        for i, hdim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.BatchNorm1d(hdim))
            layers.append(nn.ReLU())
            # Apply dropout only on last two layers
            if i >= 2:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hdim

        self.mlp = nn.Sequential(*layers)

        # 2-layer output head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.mlp(x)
        x = self.classifier(x)
        return x  # raw logits (for CrossEntropyLoss)