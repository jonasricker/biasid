from pathlib import Path
from typing import Any

import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from biasid.datasets import Metadata
from biasid.detector_wrapper import DetectorWrapper
from biasid.prediction import get_scores
from biasid.transforms import LevelTransform


def get_score_df(
    detector: DetectorWrapper,
    real_dataset: Dataset,
    fake_dataset: Dataset,
    output_dir: str | Path,
    transform: LevelTransform | None = None,
    batch_size: int = 1,
    num_workers: int = 0,
) -> pd.DataFrame:
    """
    Compute detection scores for real and fake images, clean and transformed.
    All results are saved and automatically loaded during repeated calls.
    """
    output_dir = Path(output_dir)

    rows: list[dict[str, Any]] = []

    for label, dataset in zip(["real", "fake"], [real_dataset, fake_dataset]):
        if len(dataset) == 0:
            raise ValueError(f"Dataset {dataset!r} is empty.")
        dataset_id = dataset.id if hasattr(dataset, "id") else label

        _write_status(dataset=dataset_id, detector=detector.id)
        scores, metadata = get_scores(
            detector=detector,
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            score_file=output_dir / dataset_id / "clean.pt",
        )
        rows.extend(
            _expand_prediction_rows(
                label=label,
                transform_name="clean",
                level=None,
                level_label=transform.clean_level_label if transform is not None else None,
                metadata=metadata,
                scores=scores,
            )
        )

        if transform is not None:
            for level in transform.levels:
                _write_status(dataset=dataset_id, detector=detector.id, transform=transform.id, level=level)
                scores, metadata = get_scores(
                    detector=detector,
                    dataset=dataset,
                    transform=transform.get_transform(level=level),
                    batch_size=batch_size,
                    num_workers=num_workers,
                    score_file=output_dir / dataset_id / transform.id / f"{level}.pt",
                )
                rows.extend(
                    _expand_prediction_rows(
                        label=label,
                        transform_name=transform.id,
                        level=level,
                        level_label=transform.get_level_label(level),
                        metadata=metadata,
                        scores=scores,
                    )
                )

    return pd.DataFrame(rows)


def _write_status(dataset: str, detector: str, transform: str | None = None, level: Any = None) -> None:
    transform_description = f"{transform} @ {level}" if transform is not None else "clean"
    tqdm.write(f"Predicting {dataset} ({transform_description}) with {detector}", end="")


def _expand_prediction_rows(
    scores: Tensor,
    metadata: list[Metadata],
    label: str,
    transform_name: str,
    level: Any,
    level_label: str | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for score, sample_metadata in zip(scores, metadata):
        row: dict[str, Any] = {"label": label, **sample_metadata}
        row["transform"] = transform_name
        row["level"] = level
        row["level_label"] = level_label
        row["score"] = score.item()
        rows.append(row)
    return rows
