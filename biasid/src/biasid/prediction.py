from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from tqdm.auto import tqdm

from biasid.datasets import Metadata, TransformDataset
from biasid.detector_wrapper import DetectorWrapper
from biasid.transforms import IdentityTransform


def get_scores(
    detector: DetectorWrapper,
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    score_file: Path,
    transform: Callable | None = None,
) -> tuple[Tensor, list[Metadata]]:
    """
    Load or compute scores for a single dataset and transform. Return a tuple of the scores itself and image metadata.
    """

    if not score_file.exists():
        score_file.parent.mkdir(exist_ok=True, parents=True)
        scores, metadata = compute_scores(
            detector=detector,
            dataset=dataset,
            transform=transform,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        torch.save({"scores": scores.cpu(), "metadata": metadata}, score_file)
        return scores, metadata

    tqdm.write(" -> Loading from cache.")
    data = torch.load(score_file, weights_only=False, map_location="cpu")
    scores = data["scores"]
    metadata = data["metadata"]
    return scores, metadata


@torch.inference_mode()
def compute_scores(
    detector: DetectorWrapper,
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    transform: Callable | None = None,
) -> tuple[Tensor, list[Metadata]]:
    """
    Compute scores for a single dataset and transform.
    """
    transform = transform or IdentityTransform()
    transformed_ds = TransformDataset(dataset=dataset, transform=v2.Compose([transform, detector.preprocessing]))
    dataloader = DataLoader(
        transformed_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=detector.model.custom_collate_fn if hasattr(detector.model, "custom_collate_fn") else None,
        persistent_workers=num_workers > 0,
        pin_memory=True,
    )

    all_scores: list[Tensor] = []
    all_metadata: list[Metadata] = []

    tqdm.write("")
    for batch in tqdm(dataloader, unit="batch"):
        if isinstance(batch, Sequence):
            images, metadata = batch
            metadata = [{key: value[i] for key, value in metadata.items()} for i in range(len(images))]
        else:
            images = batch
            metadata = [{} for _ in range(len(images))]

        if isinstance(images, Tensor):
            images = images.to(detector.device)
        elif isinstance(images, Sequence):
            images = [img.to(detector.device) for img in images]

        scores = detector(images)

        all_scores.append(scores)
        all_metadata.extend(metadata)

    scores = torch.cat(all_scores)
    scores = detector.postprocessing(scores)
    return scores, all_metadata
