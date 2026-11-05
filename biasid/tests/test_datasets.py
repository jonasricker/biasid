from pathlib import Path

import pytest
import torch
from PIL import Image

from biasid.datasets import FolderDataset, LabelDataset, SynthCLIC, TransformDataset


def test_label_dataset_handles_raw_and_tuple_samples(dataset):
    raw_dataset = LabelDataset(dataset.images, label=1)
    tuple_dataset = LabelDataset(dataset, label=0)

    raw_image, raw_label = raw_dataset[0]
    tuple_image, tuple_label = tuple_dataset[0]
    torch.testing.assert_close(raw_image, dataset.images[0])
    torch.testing.assert_close(tuple_image, dataset.images[0])
    assert raw_label == 1
    assert tuple_label == 0
    assert len(raw_dataset) == 16


def test_transform_dataset_applies_transform_and_preserves_metadata(dataset):
    transform = lambda value: value * 2

    metadata_dataset = TransformDataset(dataset, transform)
    raw_dataset = TransformDataset(dataset.images, transform)

    transformed_image, transformed_metadata = metadata_dataset[0]
    raw_image, raw_metadata = raw_dataset[0]
    torch.testing.assert_close(transformed_image, dataset.images[0] * 2)
    torch.testing.assert_close(raw_image, dataset.images[0] * 2)
    assert transformed_metadata == dataset.metadata[0]
    assert raw_metadata == {}


def test_folder_dataset_loads_sorted_recursive_images(tmp_path: Path):
    first_path = tmp_path / "subset-a" / "first.PNG"
    second_path = tmp_path / "subset-b" / "second.bmp"
    first_path.parent.mkdir()
    second_path.parent.mkdir()
    Image.new("L", (2, 2), 128).save(first_path)
    Image.new("RGB", (2, 2), (255, 0, 0)).save(second_path)
    (tmp_path / "ignored.txt").write_text("not an image")

    dataset = FolderDataset(tmp_path)

    assert len(dataset) == 2
    assert dataset.id == tmp_path.name
    first_image, first_metadata = dataset[0]
    second_image, second_metadata = dataset[1]
    assert first_image.shape == (3, 2, 2)
    assert second_image.shape == (3, 2, 2)
    assert first_image.dtype == torch.float32
    torch.testing.assert_close(first_image, torch.full((3, 2, 2), 128 / 255))
    torch.testing.assert_close(
        second_image,
        torch.tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        ),
    )
    assert first_metadata == {"image_id": "first", "subset": "subset-a"}
    assert second_metadata == {"image_id": "second", "subset": "subset-b"}


class FakeSynthCLICDataset:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def filter(self, function, input_columns):
        assert input_columns == "label"
        return type(self)([row for row in self.rows if function(row[input_columns])])


@pytest.mark.parametrize(
    ("label", "selected_label", "expected_id"),
    [("real", 0, "clic2020_test"), ("fake", 1, "synthclic_test")],
)
def test_synthclic_filters_label_and_converts_rows(monkeypatch, label, selected_label, expected_id):
    rows = [
        {"label": 0, "image": Image.new("L", (2, 2), 64), "image_id": "real-1", "source": "clic"},
        {"label": 1, "image": Image.new("RGB", (2, 2), (0, 255, 0)), "image_id": "fake-1", "source": "generator"},
    ]
    calls = []

    def fake_load_dataset(path, split):
        calls.append((path, split))
        return FakeSynthCLICDataset(rows)

    monkeypatch.setattr("biasid.datasets.load_dataset", fake_load_dataset)

    dataset = SynthCLIC(label=label, split="test")

    assert calls == [("marco-willi/synthclic", "test")]
    assert len(dataset) == 1
    assert dataset.id == expected_id
    image, metadata = dataset[0]
    assert image.shape == (3, 2, 2)
    assert image.dtype == torch.float32
    assert metadata["image_id"] == f"{'real' if selected_label == 0 else 'fake'}-1"
    assert metadata["subset"] == ("clic" if selected_label == 0 else "generator")


def test_synthclic_rejects_invalid_label(monkeypatch):
    monkeypatch.setattr("biasid.datasets.load_dataset", lambda **_kwargs: FakeSynthCLICDataset([]))

    with pytest.raises(ValueError, match="Invalid label"):
        SynthCLIC(label="unknown", split="test")  # type: ignore
