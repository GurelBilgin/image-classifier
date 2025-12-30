from torchvision import transforms
from PIL import Image
import torch


def prepare_image(image: Image.Image) -> torch.Tensor:
    """
    Kullanıcıdan gelen resmi model için uygun tensöre çevirir.

    Args:
        image (PIL.Image): Yüklenen görsel

    Returns:
        torch.Tensor: Model girdisi
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return transform(image).unsqueeze(0)
