"""NaimishNet CNN (Agarwal et al., 2017) for facial keypoint detection."""
import torch
import torch.nn as nn


class NaimishNet(nn.Module):
    """Four conv blocks + three FC layers for 96x96 grayscale input.

    The final layer outputs 180 values (90 keypoints x (x, y)).
    """

    def __init__(self, num_outputs: int = 180):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4), nn.ELU(),
            nn.MaxPool2d(2), nn.Dropout(0.1),
            nn.Conv2d(32, 64, kernel_size=3), nn.ELU(),
            nn.MaxPool2d(2), nn.Dropout(0.2),
            nn.Conv2d(64, 128, kernel_size=2), nn.ELU(),
            nn.MaxPool2d(2), nn.Dropout(0.3),
            nn.Conv2d(128, 256, kernel_size=1), nn.ELU(),
            nn.MaxPool2d(2), nn.Dropout(0.4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(1000), nn.ELU(), nn.Dropout(0.5),
            nn.Linear(1000, 1000), nn.Linear(1000, num_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
