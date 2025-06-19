import torch.nn as nn
import timm

class FingerprintNet(nn.Module):
    def __init__(self, backbone_model_name: str = 'resnet50', pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(backbone_model_name, pretrained=pretrained, num_classes=0)

        feature_dim = self.backbone.num_features # Timma ułatwia to

        # Głowa dla Core (x, y) - regresja
        self.core_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2) # x_core, y_core
        )

        # Głowa dla Delta (x, y) - regresja
        self.delta_coords_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2) # x_delta, y_delta
        )

        # Głowa dla detekcji istnienia Delty - klasyfikacja binarna
        self.delta_existence_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # 1 neuron dla decyzji binarnej
            # nn.Sigmoid()
        )

    def forward(self, x):
        features = self.backbone(x)

        core_coords = self.core_head(features)
        delta_coords = self.delta_coords_head(features)
        delta_existence_logits = self.delta_existence_head(features)

        return core_coords, delta_coords, delta_existence_logits