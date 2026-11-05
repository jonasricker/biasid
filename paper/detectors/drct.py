import torch
from torch import Tensor
from torchvision import transforms

from external.drct.network.models import get_models


class DRCT(torch.nn.Module):
    def __init__(self, checkpoint_path: str) -> None:
        super().__init__()

        if "convnext_base_in22k" in checkpoint_path:
            model_name = "convnext_base_in22k"
        elif "clip-ViT-L-14" in checkpoint_path:
            model_name = "clip-ViT-L-14"
        else:
            raise NotImplementedError

        self.model = get_models(model_name=model_name, embedding_size=1024)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

        self._transform = transforms.Compose(
            [
                transforms.CenterCrop(224),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def preprocessing(self, x: Tensor) -> Tensor:
        return self._transform(x)

    def postprocessing(self, x: Tensor) -> Tensor:
        return x.softmax(dim=1)[:, 1].unsqueeze(1)
