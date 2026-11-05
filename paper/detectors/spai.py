import logging

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import default_collate
from torchvision.transforms.v2.functional import to_pil_image

from external.spai.spai.config import get_config
from external.spai.spai.data.data_finetune import build_transform
from external.spai.spai.models import build_cls_model
from external.spai.spai.utils import load_pretrained

logger = logging.getLogger(__name__)


class SPAI(torch.nn.Module):
    def custom_collate_fn(self, batch):
        images, metadata = zip(*batch)
        return list(images), default_collate(metadata)

    def __init__(self, checkpoint_path: str) -> None:
        super().__init__()

        self.config = get_config({"cfg": "external/spai/configs/spai.yaml"})

        self.model = build_cls_model(self.config)
        load_pretrained(self.config, self.model, logger, checkpoint_path=checkpoint_path, verbose=False)

        self._transform = build_transform(is_train=False, config=self.config)

    def forward(self, x: tuple[Tensor]) -> Tensor:
        return self.model(x, feature_extraction_batch_size=self.config.MODEL.FEATURE_EXTRACTION_BATCH)

    def preprocessing(self, x: Tensor) -> Tensor:
        arr = np.array(to_pil_image(x))
        return self._transform(image=arr)["image"].unsqueeze(0)

    def postprocessing(self, x: Tensor) -> Tensor:
        return x.sigmoid()
