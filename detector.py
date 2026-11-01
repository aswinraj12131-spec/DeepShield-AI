import torch
import torch.nn as nn
from torchvision import models

def create_model():
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT
    )

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, 2
    )

    return model

if __name__ == "__main__":
    model = create_model()
    print("DeepShield AI model created successfully!")
    print("Classes: Fake and Real")