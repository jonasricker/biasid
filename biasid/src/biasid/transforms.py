import inspect
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from functools import partial
from io import BytesIO
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF
from PIL import Image
from torch import Tensor
from torchvision.transforms import v2


class IdentityTransform:
    def __call__(self, x):
        return x


class LevelTransform(ABC):
    """
    Base class for transforms.
    Must define a set of default levels, a unit label, and, if applicable, tick labels that replace the numerical values.
    """

    default_levels: ClassVar[Sequence[Any]]
    level_unit: ClassVar[str]
    clean_level_label: ClassVar[str] = "-"
    level_labels: ClassVar[dict[Any, str]] = {}

    def __init__(self, levels: Sequence[Any] | None = None, **kwargs: Any) -> None:
        self.levels: Sequence[Any] = tuple(levels) if levels is not None else tuple(self.default_levels)

    @property
    def id(self) -> str:
        return self.__class__.__name__.lower()

    def get_level_label(self, level: Any) -> str:
        return self.level_labels.get(level, str(level))

    @abstractmethod
    def get_transform(self, level: Any) -> Callable:
        pass


class JPEG(LevelTransform):
    default_levels = (100, 90, 80, 70, 60, 50, 40, 30, 20, 10)
    level_unit = "JPEG Quality"

    def get_transform(self, level: int) -> Callable:
        return v2.Compose(
            [
                v2.ToDtype(dtype=torch.uint8, scale=True),
                v2.JPEG(quality=level),
                v2.ToDtype(dtype=torch.float, scale=True),
            ]
        )


class WebP(LevelTransform):
    default_levels = (100, 90, 80, 70, 60, 50, 40, 30, 20, 10)
    level_unit = "WebP Quality"

    def get_transform(self, level: int) -> Callable:
        def convert(image: Tensor) -> Tensor:
            buffer = BytesIO()
            image = TF.to_dtype(image, dtype=torch.uint8, scale=True)
            TF.to_pil_image(image).save(buffer, format="webp", quality=level)

            buffer.seek(0)
            with Image.open(buffer) as pil_image:
                tensor = TF.pil_to_tensor(pil_image.convert("RGB"))

            return TF.to_dtype(tensor, dtype=torch.float, scale=True).clamp(0, 1)

        def transform(image: Tensor) -> Tensor:
            if image.ndim == 4:
                return torch.stack([convert(sample) for sample in image])
            return convert(image)

        return transform


class Grayscale(LevelTransform):
    default_levels = (1,)
    level_unit = "Color Mode"
    clean_level_label = "RGB"
    level_labels: ClassVar[dict[int, str]] = {1: "Gray"}

    def get_transform(self, level: int) -> Callable:
        if level == 1:
            return v2.Grayscale(num_output_channels=3)
        else:
            raise ValueError(f"Invalid level {level!r}. Must be 1.")


class Rotation(LevelTransform):
    default_levels = (45, 90, 135, 180, 225, 270, 315)
    level_unit = "Rotation (Degree)"
    clean_level_label = "0"

    def get_transform(self, level: int) -> Callable:
        return partial(TF.rotate, angle=level)


def factor_resize(image: Tensor, factor: float, mode) -> Tensor:
    is_single_image = image.ndim == 3
    if is_single_image:
        image = image.unsqueeze(0)

    image = F.interpolate(
        image,
        scale_factor=factor,
        mode=mode,
        antialias=True,
    ).clamp(0, 1)

    return image.squeeze(0) if is_single_image else image


class Upscaling(LevelTransform):
    default_levels = (1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0)
    level_unit = "Resize Factor"
    clean_level_label = "1.0"
    higher_means_stronger = True

    def __init__(self, levels: Sequence[float] | None = None, mode: str = "bilinear") -> None:
        super().__init__(levels=levels)
        self.mode = mode

    @property
    def id(self) -> str:
        return f"{self.__class__.__name__}_{self.mode}".lower()

    def get_transform(self, level: float) -> Callable:
        return partial(factor_resize, factor=level, mode=self.mode)


class Downscaling(LevelTransform):
    default_levels = (0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5)
    level_unit = "Resize Factor"
    clean_level_label = "1.0"

    def __init__(self, levels: Sequence[float] | None = None, mode: str = "bilinear") -> None:
        super().__init__(levels=levels)
        self.mode = mode

    @property
    def id(self) -> str:
        return f"{self.__class__.__name__}_{self.mode}".lower()

    def get_transform(self, level: float) -> Callable:
        return partial(factor_resize, factor=level, mode=self.mode)


TRANSFORM_UNITS = {}
for name, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass):
    if issubclass(cls, LevelTransform) and cls != LevelTransform:
        TRANSFORM_UNITS[cls().id.split("_")[0]] = cls.level_unit
