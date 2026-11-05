import torch
from torch import Tensor, nn

from external.aide.data.datasets import transform_train
from external.aide.data.dct import DCT_base_Rec_Module
from external.aide.models.AIDE import AIDE as OriginalAIDE


class AIDE(nn.Module):
    def __init__(self, checkpoint_path: str) -> None:
        super().__init__()

        self.model = OriginalAIDE(resnet_path=None, convnext_path=None)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["model"])

        self.dct_holder = [DCT_base_Rec_Module()]  # workaround to avoid registration as submodule

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def preprocessing(self, x: Tensor) -> Tensor:
        x_minmin, x_maxmax, x_minmin1, x_maxmax1 = self.dct_holder[0](x)

        x_0 = transform_train(x)
        x_minmin = transform_train(x_minmin)
        x_maxmax = transform_train(x_maxmax)

        x_minmin1 = transform_train(x_minmin1)
        x_maxmax1 = transform_train(x_maxmax1)

        return torch.stack([x_minmin, x_maxmax, x_minmin1, x_maxmax1, x_0], dim=0)

    def postprocessing(self, x: Tensor) -> Tensor:
        return x.softmax(dim=1)[:, 1].unsqueeze(1)
