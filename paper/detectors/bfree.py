import torch
from torch import Tensor
from torchvision import transforms

from external.bfree.code.main_bfree_single import get_config
from external.bfree.code.networks import get_network, load_weights
from external.bfree.code.utils.normalization import get_list_norm


class BFree(torch.nn.Module):
    def __init__(self, checkpoint_path: str) -> None:
        super().__init__()

        _, model_path, arch, norm_type = get_config("BFREE_dino2reg4", weights_dir=checkpoint_path)
        self.model = load_weights(get_network(arch), model_path)

        self._transform = transforms.Compose(get_list_norm(norm_type)[1:])

    def forward(self, x: tuple[Tensor]) -> Tensor:
        # B-Free does not support batch processing, so we process batch one by one
        return torch.stack([self.model(sample.unsqueeze(0))[:, 0] for sample in x])

    def preprocessing(self, x: Tensor) -> Tensor:
        return self._transform(x)

    def postprocessing(self, x: Tensor) -> Tensor:
        return x.sigmoid()
