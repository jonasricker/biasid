from collections.abc import Callable

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from biasid.detector_wrapper import DetectorWrapper
from biasid.transforms import LevelTransform


class MetadataDataset(Dataset):
    def __init__(self):
        self.images = torch.randn(16, 3, 32, 32).clamp(0, 1)
        self.metadata = [
            {"image_id": f"img_{idx}", "subset": "gen_a" if idx % 2 == 0 else "gen_b"}
            for idx in range(len(self.images))
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.metadata[index]

    @property
    def id(self):
        return "mock"


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(3 * 32 * 32, 1)

    def forward(self, images):
        return self.layer(images.flatten(start_dim=1))


@pytest.fixture
def model():
    return MockModel()


@pytest.fixture
def detector(model):
    return DetectorWrapper(model=model)


@pytest.fixture
def dataset():
    return MetadataDataset()


class MockTransform(LevelTransform):
    name = "Multiplier"
    unit = "Factor"
    default_levels = (1, 2, 3)
    higher_means_stronger = False

    def get_transform(self, level: int) -> Callable:
        return lambda x: x * level


@pytest.fixture
def transform():
    return MockTransform()
