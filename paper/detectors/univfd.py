import torch
from torch import Tensor
from torchvision import transforms

from external.univfd.models import get_model


class UnivFD(torch.nn.Module):
    def __init__(self, checkpoint_path: str) -> None:
        super().__init__()

        self.model = get_model(name="CLIP:ViT-L/14")
        self.model.fc.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

        self._transform = transforms.Compose(
            [
                transforms.CenterCrop(224),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def preprocessing(self, x: Tensor) -> Tensor:
        return self._transform(x)

    def postprocessing(self, x: Tensor) -> Tensor:
        return x.sigmoid()
