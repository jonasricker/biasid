from collections.abc import Callable
from pathlib import Path
from typing import Literal

import torch
import torchvision.transforms.v2.functional as TF
from datasets import load_dataset
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets.folder import IMG_EXTENSIONS

ScalarMetadataValue = str | int | float | bool | None
Metadata = dict[str, ScalarMetadataValue]


class FolderDataset(Dataset):
    """
    Load images from a (nested) folder.
    Metadata consists of the image's filename as 'image_id' and the parent folder name as 'subset'.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.paths = sorted([str(path) for path in self.root.rglob("*") if path.suffix.lower() in IMG_EXTENSIONS])

    @property
    def id(self) -> str:
        return self.root.name

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Tensor, Metadata]:
        path = self.paths[index]
        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")
        img = TF.to_dtype(TF.pil_to_tensor(img), dtype=torch.float, scale=True)

        return img, {"image_id": Path(path).stem, "subset": Path(path).parent.name}


class SynthCLIC(Dataset):
    """
    Load the SynthCLIC dataset (https://huggingface.co/datasets/marco-willi/synthclic).
    """

    def __init__(
        self,
        label: Literal["real", "fake"],
        split: Literal["train", "validation", "test", "all"],
    ) -> None:
        self.label = label
        self.split = split
        self.ds = load_dataset(path="marco-willi/synthclic", split=split)

        if label == "real":
            self.ds = self.ds.filter(function=lambda _label: _label == 0, input_columns="label")
        elif label == "fake":
            self.ds = self.ds.filter(function=lambda _label: _label == 1, input_columns="label")
        else:
            raise ValueError(f"Invalid label {label!r}. Must be 'real' or 'fake'.")

    @property
    def id(self) -> str:
        if self.label == "real":
            return f"clic2020_{self.split}"
        else:
            return f"synthclic_{self.split}"

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index) -> tuple[Tensor, Metadata]:
        row = self.ds[index]
        image = TF.to_dtype(TF.pil_to_tensor(row["image"].convert("RGB")), dtype=torch.float, scale=True)
        metadata = {"image_id": row["image_id"], "subset": row["source"]}

        return image, metadata


class LabelDataset(Dataset):
    """
    Assign a fixed label to a dataset. Does not pass-through metadata.
    """

    def __init__(self, dataset: Dataset, label: int) -> None:
        self.dataset = dataset
        self.label = label

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore

    def __getitem__(self, index) -> tuple[Tensor, int]:
        sample = self.dataset[index]
        if isinstance(sample, tuple):
            sample = self.dataset[index][0]
        return sample, self.label


class TransformDataset(Dataset):
    """
    Apply a transform to a dataset. Metadata is preserved.
    """

    def __init__(self, dataset: Dataset, transform: Callable) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore

    def __getitem__(self, index: int) -> tuple[Tensor, Metadata]:
        data = self.dataset[index]
        if isinstance(data, tuple):
            image, metadata = data
        else:
            image = data
            metadata = {}
        return self.transform(image), metadata
