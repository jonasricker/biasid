import re

import torch
from torch import Tensor
from torchvision import transforms

from external.rine.src.models import Model


class RINE(torch.nn.Module):
    def __init__(self, checkpoint_path: str) -> None:
        super().__init__()

        ncls = re.search(r"model_(.*)_trainable.pth", checkpoint_path).group(1).removesuffix("class")

        if ncls == "1":
            nproj = 4
            proj_dim = 1024
        elif ncls == "2":
            nproj = 4
            proj_dim = 128
        elif ncls == "4":
            nproj = 2
            proj_dim = 1024
        elif ncls == "ldm":
            nproj = 4
            proj_dim = 1024
        else:
            raise NotImplementedError

        self.model = Model(
            backbone=("ViT-L/14", 1024),
            nproj=nproj,
            proj_dim=proj_dim,
            device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        )
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        for name in state_dict:
            exec(  # noqa
                f'self.model.{name.replace(".", "[", 1).replace(".", "].", 1)} = torch.nn.Parameter(state_dict["{name}"])'
            )

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
        return self.model(x)[0]

    def preprocessing(self, x: Tensor) -> Tensor:
        return self._transform(x)

    def postprocessing(self, x: Tensor) -> Tensor:
        return x.sigmoid()
