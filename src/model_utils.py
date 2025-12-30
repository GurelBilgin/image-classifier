import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple


def load_model(
    model_path: str,
    num_classes: int,
    device: torch.device
) -> nn.Module:
    """
    Eğitilmiş ResNet18 modelini yükler.

    Args:
        model_path (str): Model ağırlıklarının yolu
        num_classes (int): Sınıf sayısı
        device (torch.device): CPU / CUDA

    Returns:
        nn.Module: Hazır model
    """
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict(
    model: nn.Module,
    input_tensor: torch.Tensor
) -> Tuple[int, float]:
    """
    Model ile tahmin yapar.

    Args:
        model (nn.Module): Eğitilmiş model
        input_tensor (torch.Tensor): [1, C, H, W]

    Returns:
        Tuple[int, float]: (tahmin edilen sınıf indexi, güven skoru)
    """
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probs, dim=1)

    return prediction.item(), confidence.item()
