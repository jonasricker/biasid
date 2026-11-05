import pandas as pd
import pytest
import torch
from torchmetrics.functional.classification import (
    binary_accuracy,
    binary_auroc,
    binary_average_precision,
)

from biasid.metrics import (
    BootstrapResult,
    _average_over_fake_subsets,
    batched_binary_accuracy,
    batched_binary_auroc,
    batched_binary_average_precision,
    compute_ci,
)


def test_average_over_fake_subsets_averages_bootstrap_results():
    score_df = pd.DataFrame(
        {
            "label": ["fake", "fake", "fake", "real"],
            "subset": ["fake_a", "fake_b", "fake_c", "real_a"],
            "transform": ["blur", "blur", "blur", "blur"],
            "level": [1, 1, 1, 1],
            "score": [
                BootstrapResult(torch.tensor(0.2), torch.tensor([0.0, 0.2, 0.4]), torch.tensor(0.0), torch.tensor(0.4)),
                BootstrapResult(torch.tensor(0.6), torch.tensor([0.4, 0.6, 0.8]), torch.tensor(0.4), torch.tensor(0.8)),
                BootstrapResult(torch.tensor(0.9), torch.tensor([0.7, 0.9, 1.1]), torch.tensor(0.7), torch.tensor(1.1)),
                BootstrapResult(torch.tensor(100.0), torch.tensor([100.0]), torch.tensor(100.0), torch.tensor(100.0)),
            ],
        }
    )

    result = _average_over_fake_subsets(score_df, metric_col="score")
    average = result.iloc[0]["score"]
    fake_metrics = score_df.loc[score_df["label"] == "fake", "score"]
    expected_estimate = sum(metric.estimate.item() for metric in fake_metrics) / len(fake_metrics)
    expected_statistics = torch.stack([metric.bootstrap_statistics for metric in fake_metrics]).mean(dim=0)
    expected_ci_lower, expected_ci_upper = compute_ci(expected_statistics)

    torch.testing.assert_close(average.estimate, torch.tensor(expected_estimate, dtype=torch.float64))
    torch.testing.assert_close(average.bootstrap_statistics, expected_statistics)
    torch.testing.assert_close(average.ci_lower, expected_ci_lower)
    torch.testing.assert_close(average.ci_upper, expected_ci_upper)
    assert result.iloc[0]["label"] == "fake"
    assert result.iloc[0]["subset"] == "all"


def test_average_over_fake_subsets_keeps_metadata_groups_separate():
    metric = BootstrapResult(torch.tensor(1.0), torch.tensor([1.0, 1.0]), torch.tensor(1.0), torch.tensor(1.0))
    score_df = pd.DataFrame(
        {
            "label": ["fake", "fake", "fake"],
            "subset": ["fake_a", "fake_b", "fake_a"],
            "transform": ["blur", "blur", "noise"],
            "level": [1, 1, 1],
            "score": [metric, metric, metric],
        }
    )

    result = _average_over_fake_subsets(score_df, metric_col="score")

    assert result[["transform", "level", "subset"]].to_dict("records") == [
        {"transform": "blur", "level": 1, "subset": "all"},
        {"transform": "noise", "level": 1, "subset": "all"},
    ]


def test_average_over_fake_subsets_keeps_real_subsets_separate():
    score_df = pd.DataFrame(
        {
            "real_subset": ["real_a", "real_a", "real_b", "real_b"],
            "fake_subset": ["fake_a", "fake_b", "fake_a", "fake_b"],
            "transform": ["blur"] * 4,
            "level": [1] * 4,
            "score": [
                BootstrapResult(torch.tensor(0.2), torch.tensor([0.2]), torch.tensor(0.2), torch.tensor(0.2)),
                BootstrapResult(torch.tensor(0.4), torch.tensor([0.4]), torch.tensor(0.4), torch.tensor(0.4)),
                BootstrapResult(torch.tensor(0.6), torch.tensor([0.6]), torch.tensor(0.6), torch.tensor(0.6)),
                BootstrapResult(torch.tensor(0.8), torch.tensor([0.8]), torch.tensor(0.8), torch.tensor(0.8)),
            ],
        }
    )

    result = _average_over_fake_subsets(score_df, metric_col="score")

    assert result[["real_subset", "fake_subset"]].to_dict("records") == [
        {"real_subset": "real_a", "fake_subset": "all"},
        {"real_subset": "real_b", "fake_subset": "all"},
    ]
    torch.testing.assert_close(result.iloc[0]["score"].estimate, torch.tensor(0.3, dtype=torch.float64))
    torch.testing.assert_close(result.iloc[1]["score"].estimate, torch.tensor(0.7, dtype=torch.float64))


@pytest.fixture
def score_batches() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    real_scores = torch.rand((10, 100), generator=generator)
    fake_scores = torch.rand((10, 100), generator=generator)

    real_scores[:, :10] = 0.5
    fake_scores[:, :10] = 0.5
    real_scores[:, 10:20] = torch.tensor([0.499999, 0.5, 0.500001, 0.0, 1.0] * 2)
    fake_scores[:, 10:20] = torch.tensor([0.500001, 0.5, 0.499999, 1.0, 0.0] * 2)

    return real_scores, fake_scores


def test_auroc_matches_torchmetrics(score_batches):
    real_scores, fake_scores = score_batches

    expected = torch.stack(
        [
            binary_auroc(
                torch.cat([real_row, fake_row]),
                torch.cat([torch.zeros_like(real_row), torch.ones_like(fake_row)]).int(),
            )
            for real_row, fake_row in zip(real_scores, fake_scores)
        ]
    )

    torch.testing.assert_close(batched_binary_auroc(real_scores, fake_scores), expected)


def test_binary_accuracy_matches_torchmetrics(score_batches):
    real_scores, fake_scores = score_batches

    expected = torch.stack(
        [
            binary_accuracy(
                torch.cat([real_row, fake_row]),
                torch.cat([torch.zeros_like(real_row), torch.ones_like(fake_row)]).int(),
            )
            for real_row, fake_row in zip(real_scores, fake_scores)
        ]
    )

    torch.testing.assert_close(batched_binary_accuracy(real_scores, fake_scores), expected)


def test_binary_average_precision_matches_torchmetrics(score_batches):
    real_scores, fake_scores = score_batches

    expected = torch.stack(
        [
            binary_average_precision(
                torch.cat([real_row, fake_row]),
                torch.cat([torch.zeros_like(real_row), torch.ones_like(fake_row)]).int(),
            )
            for real_row, fake_row in zip(real_scores, fake_scores)
        ]
    )

    torch.testing.assert_close(batched_binary_average_precision(real_scores, fake_scores), expected)
