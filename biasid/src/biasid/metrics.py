from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import Tensor
from tqdm.auto import tqdm


@dataclass
class MetricsResult:
    """
    Container for all metrics, each represented as a DataFrame.
    Supports saving and loading, with each DataFrame saved as an individual CSV file.
    """

    mean_score: pd.DataFrame
    mss: pd.DataFrame
    class_ats: pd.DataFrame
    overall_ats: pd.DataFrame
    clf: pd.DataFrame

    def save(self, save_dir: str | Path, *args, **kwargs):
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        for name in ("mean_score", "mss", "class_ats", "overall_ats", "clf"):
            getattr(self, name).to_csv(save_dir / f"{name}.csv", *args, **kwargs)

    @classmethod
    def load(cls, save_dir: str | Path, *args, **kwargs):
        save_dir = Path(save_dir)
        names = ("mean_score", "mss", "class_ats", "overall_ats", "clf")
        dataframes = {name: pd.read_csv(save_dir / f"{name}.csv", *args, **kwargs) for name in names}
        return cls(**dataframes)


@dataclass
class BootstrapResult:
    """
    Container for bootstrapping, containing estimate, result for all resamples, and confidence interval.
    """

    estimate: Tensor
    bootstrap_statistics: Tensor
    ci_lower: Tensor
    ci_upper: Tensor


def compute_metrics(score_df: pd.DataFrame, B: int = 10_000) -> MetricsResult:
    """
    Compute mean scores, bias metrics, and classification metrics.
    Performs bootstrapping with 'B' resamples to compute confidence intervals.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pbar = tqdm(
        desc="Computing metrics",
        total=2 * score_df.groupby(["label", "subset", "transform", "level"], dropna=False).ngroups
        + score_df.groupby(["label", "subset"], dropna=False).ngroups
        + (score_df.groupby(["transform", "level"], dropna=False).ngroups + 1)
        * score_df.query("label == 'real'")["subset"].nunique()
        * score_df.query("label == 'fake'")["subset"].nunique(),
    )

    # compute mean score
    mean_df = []
    group_cols = ["label", "subset", "transform", "level"]
    for (label, subset, transform, level), group_df in score_df.groupby(group_cols, dropna=False, sort=False):
        score = torch.tensor(group_df["score"].values, device=device)
        mean_score = bootstrap(
            inputs={"input": score},
            statistic=partial(torch.mean, dim=-1),
            B=B,
        )
        mean_df.append(
            {
                "label": label,
                "subset": subset,
                "transform": transform,
                "level": level,
                "level_label": group_df["level_label"].iloc[0],
                "mean_score": mean_score,
            }
        )
        pbar.update()

    mean_df = pd.DataFrame(mean_df)

    # compute mean score shift
    mss_df = []
    group_cols = ["label", "subset", "transform", "level"]
    for (label, subset, transform, level), group_df in score_df.groupby(group_cols, dropna=False, sort=False):
        clean = torch.tensor(
            score_df.query("label == @label and subset == @subset and transform == 'clean'")["score"].values,
            device=device,
        )
        transformed = torch.tensor(group_df["score"].values, device=device)
        mss = bootstrap(inputs={"clean": clean, "transformed": transformed}, statistic=compute_mss, B=B)

        mss_df.append(
            {
                "label": label,
                "subset": subset,
                "transform": transform,
                "level": level,
                "level_label": group_df["level_label"].iloc[0],
                "mss": mss,
            }
        )
        pbar.update()

    mss_df = pd.DataFrame(mss_df)

    # compute per-class ATS
    class_ats_df = []
    group_cols = ["label", "subset"]
    for (label, subset), group_df in score_df.groupby(group_cols, dropna=False, sort=False):
        clean = torch.tensor(group_df.query("transform == 'clean'")["score"].values, device=device)
        transformed = {
            level: torch.tensor(level_df["score"].values, device=device)
            for level, level_df in group_df.query("transform != 'clean'").groupby("level")
        }
        class_ats = bootstrap(inputs={"clean": clean, "transformed": transformed}, statistic=compute_class_ats, B=B)

        class_ats_df.append(
            {
                "label": label,
                "subset": subset,
                "transform": group_df.query("transform != 'clean'")["transform"].iloc[0],
                "class_ats": class_ats,
            }
        )
        pbar.update()

    class_ats_df = pd.DataFrame(class_ats_df)

    # compute overall ATS
    overall_ats_df = []
    for real_subset, real_df in score_df.query("label == 'real'").groupby("subset", sort=False):
        real_clean = torch.tensor(real_df.query("transform == 'clean'")["score"].values, device=device)
        for fake_subset, fake_df in score_df.query("label == 'fake'").groupby("subset", sort=False):
            fake_clean = torch.tensor(fake_df.query("transform == 'clean'")["score"].values, device=device)
            real_transformed = {
                level: torch.tensor(level_df["score"].values, device=device)
                for level, level_df in real_df.query("transform != 'clean'").groupby("level")
            }
            fake_transformed = {
                level: torch.tensor(level_df["score"].values, device=device)
                for level, level_df in fake_df.query("transform != 'clean'").groupby("level")
            }

            overall_ats = bootstrap(
                inputs={
                    "real_clean": real_clean,
                    "fake_clean": fake_clean,
                    "real_transformed": real_transformed,
                    "fake_transformed": fake_transformed,
                },
                statistic=compute_overall_ats,
                B=B,
            )
            overall_ats_df.append(
                {
                    "real_subset": real_subset,
                    "fake_subset": fake_subset,
                    "transform": real_df.query("transform != 'clean'")["transform"].iloc[0],
                    "overall_ats": overall_ats,
                }
            )
            pbar.update()

    overall_ats_df = pd.DataFrame(overall_ats_df)

    # compute classification metrics
    clf_df = []
    for (transform, level), group_df in score_df.groupby(["transform", "level"], dropna=False, sort=False):
        for real_subset, real_df in group_df.query("label == 'real'").groupby("subset", sort=False):
            real = torch.tensor(real_df["score"].values, device=device)
            for fake_subset, fake_df in group_df.query("label == 'fake'").groupby("subset", sort=False):
                fake = torch.tensor(fake_df["score"].values, device=device)

                auroc = bootstrap(
                    inputs={"real_scores": real, "fake_scores": fake}, statistic=batched_binary_auroc, B=B
                )
                ap = bootstrap(
                    inputs={"real_scores": real, "fake_scores": fake}, statistic=batched_binary_average_precision, B=B
                )
                acc = bootstrap(
                    inputs={"real_scores": real, "fake_scores": fake}, statistic=batched_binary_accuracy, B=B
                )

                clf_df.append(
                    {
                        "real_subset": real_subset,
                        "fake_subset": fake_subset,
                        "transform": transform,
                        "level": level,
                        "level_label": group_df["level_label"].iloc[0],
                        "auroc": auroc,
                        "ap": ap,
                        "acc": acc,
                    }
                )
                pbar.update()

    clf_df = pd.DataFrame(clf_df)

    # compute average over all fake subsets
    mean_df = pd.concat([mean_df, _average_over_fake_subsets(df=mean_df, metric_col="mean_score")], ignore_index=True)
    mss_df = pd.concat([mss_df, _average_over_fake_subsets(df=mss_df, metric_col="mss")], ignore_index=True)
    class_ats_df = pd.concat(
        [class_ats_df, _average_over_fake_subsets(df=class_ats_df, metric_col="class_ats")],
        ignore_index=True,
    )
    overall_ats_df = pd.concat(
        [overall_ats_df, _average_over_fake_subsets(df=overall_ats_df, metric_col="overall_ats")], ignore_index=True
    )
    clf_df = pd.concat(
        [clf_df, _average_over_fake_subsets(df=clf_df, metric_col=["auroc", "ap", "acc"])], ignore_index=True
    )

    # convert BootstrapResults to regular columns
    mean_df = _expand_bootstrap_results(df=mean_df, metric_col="mean_score")
    mss_df = _expand_bootstrap_results(df=mss_df, metric_col="mss")
    class_ats_df = _expand_bootstrap_results(df=class_ats_df, metric_col="class_ats")
    overall_ats_df = _expand_bootstrap_results(df=overall_ats_df, metric_col="overall_ats")
    clf_df = _expand_bootstrap_results(df=clf_df, metric_col=["auroc", "ap", "acc"])

    return MetricsResult(
        mean_score=mean_df,
        mss=mss_df,
        class_ats=class_ats_df,
        overall_ats=overall_ats_df,
        clf=clf_df,
    )


def bootstrap(
    inputs: dict[str, Tensor | dict[Any, Tensor]],
    statistic: Callable[..., Tensor],
    B: int = 10_000,
    seed: int = 0,
) -> BootstrapResult:
    """
    Compute statistic with bootstrapping.
    """
    first_input = next(iter(inputs.values()))
    first_tensor = first_input if isinstance(first_input, Tensor) else next(iter(first_input.values()))
    sample_count = first_tensor.shape[0]
    estimate = statistic(**inputs)

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        sample_count,
        (B, sample_count),
        generator=generator,
    )

    sampled = {name: _sample_input(input_data, indices) for name, input_data in inputs.items()}
    bootstrap_statistics = statistic(**sampled)

    ci_lower, ci_upper = compute_ci(bootstrap_statistics, confidence_level=0.95)

    return BootstrapResult(
        estimate=estimate,
        bootstrap_statistics=bootstrap_statistics,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )


def compute_ci(
    bootstrap_statistics: Tensor | Sequence[Tensor],
    confidence_level: float = 0.95,
) -> tuple[Tensor, Tensor]:
    """
    Compute confidence interval from bootstrap result.
    """
    bootstrap_statistics = (
        bootstrap_statistics
        if isinstance(bootstrap_statistics, Tensor)
        else torch.stack(tuple(bootstrap_statistics)).mean(dim=0)
    )
    tail_probability = (1 - confidence_level) / 2
    bounds = torch.quantile(
        input=bootstrap_statistics,
        q=torch.tensor(
            [tail_probability, 1 - tail_probability],
            dtype=bootstrap_statistics.dtype,
            device=bootstrap_statistics.device,
        ),
        dim=0,
    )
    return bounds[0], bounds[1]


def compute_mss(clean: Tensor, transformed: Tensor) -> Tensor:
    """
    Compute mean score shift (MSS) between scores on clean and transformed (for a single level) images.
    """
    return (transformed - clean).mean(dim=-1)


def compute_class_ats(clean: Tensor, transformed: dict[Any, Tensor]) -> Tensor:
    """
    Compute aggregated transformation sensitivity (ATS) for a single class.
    """
    per_level_mss = [compute_mss(clean, transformed) for transformed in transformed.values()]
    return torch.stack(per_level_mss).mean(dim=0)


def compute_overall_ats(
    real_clean: Tensor,
    fake_clean: Tensor,
    real_transformed: dict[Any, Tensor],
    fake_transformed: dict[Any, Tensor],
) -> Tensor:
    """
    Compute overall transformation sensitivity (ATS), combining both real and fake scores.
    """
    real_transformed = dict(sorted(real_transformed.items()))
    fake_transformed = dict(sorted(fake_transformed.items()))
    if real_transformed.keys() != fake_transformed.keys():
        raise ValueError("real_transformed and fake_transformed must have the same levels")

    per_level_mss_real = [compute_mss(real_clean, transformed) for transformed in real_transformed.values()]
    per_level_mss_fake = [compute_mss(fake_clean, transformed) for transformed in fake_transformed.values()]

    return (torch.stack(per_level_mss_real) + torch.stack(per_level_mss_fake)).abs().mean(dim=0)


def batched_binary_accuracy(real_scores: Tensor, fake_scores: Tensor) -> Tensor:
    """
    Batched implementation of binary accuracy, assuming a decision threshold of 0.5.
    """
    y_true = torch.cat([torch.zeros_like(real_scores), torch.ones_like(fake_scores)], dim=-1).int()
    y_pred = (torch.cat([real_scores, fake_scores], dim=-1) > 0.5).int()
    return (y_pred == y_true).float().mean(dim=-1)


def batched_binary_auroc(real_scores: Tensor, fake_scores: Tensor) -> Tensor:
    """
    Batched implementation of binary AUROC.
    """
    y_true = torch.cat([torch.zeros_like(real_scores), torch.ones_like(fake_scores)], dim=-1).int()
    y_score = torch.cat([real_scores, fake_scores], dim=-1)

    if real_scores.ndim == 1:
        y_true = y_true.unsqueeze(0)
        y_score = y_score.unsqueeze(0)

    B, N = y_true.shape

    # Sort scores in ascending order, independently for each task.
    sorted_scores, indices = torch.sort(y_score, dim=-1)
    labels = y_true.gather(dim=-1, index=indices).to(torch.float)

    # Find tied adjacent scores.
    tied = sorted_scores[:, 1:] == sorted_scores[:, :-1]

    positions = torch.arange(N, device=y_score.device).expand(B, N)

    # Start and end positions of each tie group.
    group_start = torch.cat(
        [
            torch.ones((B, 1), dtype=torch.bool, device=y_score.device),
            ~tied,
        ],
        dim=1,
    )

    group_end = torch.cat(
        [
            ~tied,
            torch.ones((B, 1), dtype=torch.bool, device=y_score.device),
        ],
        dim=1,
    )

    starts = torch.where(group_start, positions, 0).cummax(dim=-1).values

    ends = torch.flip(
        torch.where(group_end, positions, N - 1).flip(dims=[1]).cummin(dim=-1).values,
        dims=[-1],
    )

    # Average 1-based ranks for tied scores.
    ranks = (starts.to(torch.float) + ends.to(torch.float) + 2) / 2

    # All reductions are over dim=1, leaving one result per task.
    n_pos = labels.sum(dim=-1)  # [B]
    n_neg = N - n_pos  # [B]
    rank_sum = (ranks * labels).sum(dim=1)  # [B]

    auroc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)  # [B]

    return auroc if real_scores.ndim > 1 else auroc.squeeze(0)


def batched_binary_average_precision(real_scores: Tensor, fake_scores: Tensor) -> Tensor:
    """
    Batched implementation of binary average precision (AP).
    """
    y_true = torch.cat([torch.zeros_like(real_scores), torch.ones_like(fake_scores)], dim=-1).int()
    y_score = torch.cat([real_scores, fake_scores], dim=-1)

    if real_scores.ndim == 1:
        y_true = y_true.unsqueeze(0)
        y_score = y_score.unsqueeze(0)

    B, N = y_true.shape

    y_true = y_true.bool()

    # Sort scores in descending order.
    indices = torch.argsort(y_score, dim=1, descending=True)
    scores = y_score.gather(1, indices)
    labels = y_true.gather(1, indices).to(torch.float)

    # Cumulative true positives.
    cum_tp = labels.cumsum(dim=1)

    # Number of samples considered at each rank.
    ranks = torch.arange(1, N + 1, device=y_score.device, dtype=torch.float).unsqueeze(0)

    # Precision at each rank.
    precision = cum_tp / ranks

    # A threshold ends at the last sample with a given score.
    is_end = torch.ones((B, N), dtype=torch.bool, device=y_score.device)
    is_end[:, :-1] = scores[:, :-1] != scores[:, 1:]

    # Identify the beginning of each group of tied scores.
    is_start = torch.ones_like(is_end)
    is_start[:, 1:] = scores[:, 1:] != scores[:, :-1]

    # Cumulative true positives immediately before each tie group.
    prev_tp = torch.cat(
        [
            torch.zeros((B, 1), dtype=torch.float, device=y_score.device),
            cum_tp[:, :-1],
        ],
        dim=1,
    )

    # Propagate the preceding group's cumulative TP count.
    prev_tp = torch.where(is_start, prev_tp, 0.0)
    prev_tp = prev_tp.cummax(dim=1).values

    # Positive samples contributed by each distinct threshold.
    delta_tp = (cum_tp - prev_tp) * is_end

    # Total positives per task.
    n_pos = labels.sum(dim=1)

    # Average precision = sum(precision * recall increment).
    ap = (precision * delta_tp).sum(dim=1) / n_pos.clamp(min=1)

    # Convention for tasks without positive samples.
    ap = torch.where(n_pos > 0, ap, 0.0)

    return ap if real_scores.ndim > 1 else ap.squeeze(0)


def _average_over_fake_subsets(df: pd.DataFrame, metric_col: str | list[str]) -> pd.DataFrame:
    def bootstrap_average(s: pd.Series) -> BootstrapResult:
        mean_estimate = s.apply(lambda x: x.estimate).mean()
        mean_statistics = torch.stack(s.apply(lambda x: x.bootstrap_statistics).tolist()).mean(dim=0)
        ci_lower, ci_upper = compute_ci(mean_statistics, confidence_level=0.95)

        return BootstrapResult(
            estimate=torch.tensor(mean_estimate),
            bootstrap_statistics=mean_statistics,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
        )

    if isinstance(metric_col, str):
        metric_col = [metric_col]

    fake_df = df.query("label == 'fake'") if "label" in df.columns else df
    subset_col = {"subset", "fake_subset"} & set(fake_df.columns)
    group_cols = [col for col in fake_df.columns if col not in subset_col | set(metric_col)]
    average_df = (
        fake_df.groupby(group_cols, dropna=False, sort=False)
        .apply(lambda group: pd.Series({col: bootstrap_average(group[col]) for col in metric_col}))
        .reset_index()
    )
    average_df["subset" if "subset" in df.columns else "fake_subset"] = "all"
    return average_df


def _expand_bootstrap_results(df: pd.DataFrame, metric_col: str | list[str]) -> pd.DataFrame:
    if isinstance(metric_col, str):
        metric_col = [metric_col]

    for col in metric_col:
        estimates = df[col].map(lambda b: b.estimate.item())  # type: ignore
        ci_lowers = df[col].map(lambda b: b.ci_lower.item())  # type: ignore
        ci_uppers = df[col].map(lambda b: b.ci_upper.item())  # type: ignore
        df[col] = estimates
        df[f"{col}_ci_lower"] = ci_lowers
        df[f"{col}_ci_upper"] = ci_uppers

    return df


def _sample_input(data: Tensor | dict[Any, Tensor], indices: Tensor) -> Tensor | dict[Any, Tensor]:
    if isinstance(data, Tensor):
        return data[indices]
    return {key: value[indices] for key, value in data.items()}


METRIC_LABELS = {"score": "Score", "mss": "MSS", "acc": "Acc.", "ap": "AP", "auroc": "AUC"}
