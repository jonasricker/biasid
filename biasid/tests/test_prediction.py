import torch

from biasid.detector_wrapper import DetectorWrapper
from biasid.prediction import compute_scores, get_scores


def test_wrapper_id(detector):
    assert detector.id == "mockmodel"


def test_compute_scores(model, detector, dataset):
    preds, metadata = compute_scores(detector=detector, dataset=dataset, batch_size=2, num_workers=0)

    torch.testing.assert_close(preds, model(dataset.images))
    assert metadata == dataset.metadata


def test_transform(model, detector, dataset, transform):
    preds, _ = compute_scores(
        detector=detector, dataset=dataset, batch_size=4, num_workers=0, transform=transform.get_transform(level=0.5)
    )

    torch.testing.assert_close(preds, model(dataset.images / 2))


def test_preprocessing(model, dataset):
    detector = DetectorWrapper(model, preprocessing=lambda x: x / 2)
    preds, _ = compute_scores(detector=detector, dataset=dataset, batch_size=4, num_workers=0)

    torch.testing.assert_close(preds, model(dataset.images / 2))


def test_transform_and_preprocessing(model, dataset, transform):
    detector = DetectorWrapper(model, preprocessing=lambda x: x / 2)

    preds, _ = compute_scores(
        detector=detector, dataset=dataset, batch_size=4, num_workers=0, transform=transform.get_transform(level=0.5)
    )

    torch.testing.assert_close(preds, model(dataset.images / 4))


def test_postprocessing(model, dataset):
    detector = DetectorWrapper(model, postprocessing=lambda x: x / 2)
    preds, _ = compute_scores(detector=detector, dataset=dataset, batch_size=4, num_workers=0)

    torch.testing.assert_close(preds, model(dataset.images) / 2)


def test_get_scores_roundtrip(tmp_path, model, detector, dataset):
    scores, metadata = get_scores(
        detector=detector,
        dataset=dataset,
        transform=None,
        batch_size=4,
        num_workers=0,
        score_file=tmp_path / "scores.pt",
    )

    assert len(scores) == len(dataset.metadata)
    assert metadata == dataset.metadata

    scores_reload, metadata_reload = get_scores(
        detector=detector,
        dataset=dataset,
        transform=None,
        batch_size=4,
        num_workers=0,
        score_file=tmp_path / "scores.pt",
    )

    torch.testing.assert_close(scores_reload, scores)
    assert metadata_reload == metadata


def test_missing_metadata(dataset, model):
    detector = DetectorWrapper(model)

    class PlainDataset(dataset.__class__):
        def __getitem__(self, index):
            image, _ = super().__getitem__(index)
            return image

    _scores, metadata = compute_scores(detector=detector, dataset=PlainDataset(), batch_size=4, num_workers=0)
    assert metadata == [{} for _ in range(len(dataset))]
