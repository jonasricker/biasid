from pathlib import Path

import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from biasid.metrics import METRIC_LABELS, MetricsResult
from biasid.transforms import TRANSFORM_UNITS
from biasid.visualization import _configure_mpl, _plot_lines, _set_figsize
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score

DETECTOR_LABELS = {
    "univfd": "UnivFD",
    "drct": "DRCT",
    "rine": "RINE",
    "aide": "AIDE",
    "spai": "SPAI",
    "bfree": "B-Free",
}

DATASET_LABELS = {
    "raise1k_resized": "Synthbuster",
    "raise1k": "Synthbuster (original RAISE-1k)",
    "clic2020_all": "SynthCLIC",
}

ATS_LABELS = {
    "real_ats": r"$\text{ATS}(X_R)$",
    "fake_ats": r"$\text{ATS}(X_F)$",
    "overall_ats": r"$\text{ATS}(X_R, X_F)$",
}


def format_float(inpt: float) -> str:
    out = f"{inpt:.2f}"
    if out == "1.00":
        return "1.0"
    if out.startswith("-0."):
        return "-" + out[2:]
    return out.lstrip("0")


def create_plot_value_df(
    metrics: MetricsResult, real_dataset_id: str, fake_dataset_id: str, detector_id: str
) -> pd.DataFrame:
    real_scores = metrics.mean_score.query("label == 'real'").set_index("level_label")["mean_score"]
    fake_scores = metrics.mean_score.query("label == 'fake' and subset == 'all'").set_index("level_label")["mean_score"]
    real_mss = metrics.mss.query("label == 'real'").set_index("level_label")["mss"]
    fake_mss = metrics.mss.query("label == 'fake' and subset == 'all'").set_index("level_label")["mss"]
    auroc = metrics.clf.query("fake_subset == 'all'").set_index("level_label")["auroc"]

    detector_label = DETECTOR_LABELS.get(detector_id, detector_id)

    df = pd.concat(
        {
            (detector_label, "score", "real"): real_scores,
            (detector_label, "score", "fake"): fake_scores,
            (detector_label, "mss", "real"): real_mss,
            (detector_label, "mss", "fake"): fake_mss,
            (detector_label, METRIC_LABELS["auroc"], ""): auroc,
        },
        axis=1,
    )

    df.columns = pd.MultiIndex.from_tuples(
        df.columns,
        names=["detector", "metric", "label"],
    )

    transform = metrics.class_ats.iloc[0]["transform"]
    df.index = pd.MultiIndex.from_product(
        [[f"{real_dataset_id}_vs_{fake_dataset_id}"], [transform], df.index],
        names=["setting", "transform", "level"],
    )

    return df


def create_ats_row(
    metrics: MetricsResult,
    real_dataset_id: str,
    fake_dataset_id: str,
    detector_id: str,
    n_samples: int | None = None,
    ci_gray: bool = True,
) -> pd.DataFrame:
    def _format_mean_and_ci(result: pd.Series, metric: str) -> tuple[str, str, str]:
        value = f"{result[metric]:.3f}"
        if ci_gray:
            ci = f"\\textcolor{{gray}}{{\\fontsize{{5}}{{5}}\\selectfont ({result[f'{metric}_ci_lower']:.2f}, {result[f'{metric}_ci_upper']:.2f})}}"
        else:
            ci = f"({result[f'{metric}_ci_lower']:.3f}, {result[f'{metric}_ci_upper']:.3f})"
        ci_width = f"{result[f'{metric}_ci_upper'] - result[f'{metric}_ci_lower']:.3f}"
        return value, ci, ci_width

    real_ats = metrics.class_ats.query("label == 'real'").iloc[0]
    fake_ats = metrics.class_ats.query("label == 'fake' and subset == 'all'").iloc[0]
    overall_ats = metrics.overall_ats.query("fake_subset == 'all'").iloc[0]

    clean_clf = metrics.clf.query("fake_subset == 'all' and transform == 'clean'")
    transformed_clf = metrics.clf.query("fake_subset == 'all' and transform != 'clean'")
    mean_clf_shift = (transformed_clf.loc[:, ["acc", "auroc"]] - clean_clf.loc[:, ["acc", "auroc"]].iloc[0]).mean()

    real_value, real_ci, real_ci_width = _format_mean_and_ci(real_ats, "class_ats")
    fake_value, fake_ci, fake_ci_width = _format_mean_and_ci(fake_ats, "class_ats")
    overall_value, overall_ci, overall_ci_width = _format_mean_and_ci(overall_ats, "overall_ats")

    table = {
        "detector": [DETECTOR_LABELS[detector_id]],
        "dataset": [DATASET_LABELS[real_dataset_id]],
        "transform": [real_ats["transform"]],
        ATS_LABELS["real_ats"]: [real_value],
        ATS_LABELS["real_ats"] + "_ci": [real_ci],
        ATS_LABELS["real_ats"] + "_ci_width": [real_ci_width],
        ATS_LABELS["fake_ats"]: [fake_value],
        ATS_LABELS["fake_ats"] + "_ci": [fake_ci],
        ATS_LABELS["fake_ats"] + "_ci_width": [fake_ci_width],
        ATS_LABELS["overall_ats"]: [overall_value],
        ATS_LABELS["overall_ats"] + "_ci": [overall_ci],
        ATS_LABELS["overall_ats"] + "_ci_width": [overall_ci_width],
        "Acc. Shift": [f"{mean_clf_shift['acc']:.3f}"],
        "AUC Shift": [f"{mean_clf_shift['auroc']:.3f}"],
    }
    if n_samples is not None:
        table["n_samples"] = n_samples

    return pd.DataFrame(table)


def render_bias_analysis_table(df: pd.DataFrame, path: str | Path) -> None:
    df = (
        df.drop("transform", axis=1)
        .pivot(
            index=["detector"],
            columns="dataset",
        )
        .swaplevel(0, 1, axis=1)
        .sort_index(axis=1, level=0, ascending=False)
    )
    df = df.reindex(list(DATASET_LABELS.values()), axis=1, level=0)
    metric_order = [metric for ats in ATS_LABELS.values() for metric in (ats, f"{ats}_ci")] + [
        "Acc. Shift",
        "AUC Shift",
    ]
    df = df.reindex(metric_order, axis=1, level=1)
    df = df.reindex(list(DETECTOR_LABELS.values()))

    df.index.name = None
    df.columns.names = [None] * df.columns.nlevels

    df.to_latex(path, multicolumn_format="c")


def plot_score_and_mss(
    result: MetricsResult,
    real_subset: str,
    fake_subset: str,
) -> tuple[Figure, Axes]:
    real_query = "label == 'real' and subset == @real_subset"
    fake_query = "label == 'fake' and subset == @fake_subset"

    real_mean_score_df = result.mean_score.query(real_query)
    fake_mean_score_df = result.mean_score.query(fake_query)
    real_mss_df = result.mss.query(real_query)
    fake_mss_df = result.mss.query(fake_query)
    transform = real_mean_score_df.query("transform != 'clean'")["transform"].iloc[0]

    _configure_mpl()
    _set_figsize(factor=(0.15, 0.24))
    plt.rcParams.update({"axes.labelpad": 0.5})
    fig, axs = plt.subplots(nrows=2, sharex=True)
    fig.align_ylabels()

    # plot mean score
    _plot_lines(first=real_mean_score_df, second=fake_mean_score_df, metric="mean_score", ax=axs[0])
    axs[0].set_ylim(-0.09, 1.09)
    axs[0].set_ylabel("Score")
    axs[0].yaxis.set_major_locator(mticker.FixedLocator([0, 0.5, 1]))
    axs[0].yaxis.set_major_formatter(mticker.FixedFormatter(["0", "", "1"]))

    _plot_lines(first=real_mss_df, second=fake_mss_df, metric="mss", ax=axs[1])
    axs[1].set_ylim(-1.18, 1.18)
    axs[1].set_ylabel("MSS")
    axs[1].yaxis.set_major_locator(mticker.FixedLocator([-1, -0.5, 0, 0.5, 1]))
    axs[1].yaxis.set_major_formatter(mticker.FixedFormatter(["-1", "", "0", "", "1"]))

    axs[-1].set_xlabel(TRANSFORM_UNITS.get(transform.split("_")[0], transform))

    # only show every second label to avoid clutter
    xlabels = axs[-1].get_xticklabels()
    if len(xlabels) > 6:
        for index, label in enumerate(xlabels):
            label.set_visible(index % 2 == 0)

    if len(real_mean_score_df) < 4:
        axs[-1].set_xlim(axs[2].get_xlim()[0] - 0.1, axs[-1].get_xlim()[1] + 0.1)

    return fig, axs


def plot_accuracy_heatmap(
    score_df: pd.DataFrame,
    cbar: bool = True,
) -> tuple[Figure, Axes]:
    _configure_mpl()
    plt.rcParams.update({"axes.labelpad": 0.5})
    _set_figsize(factor=(0.24, 0.24))

    # compute accuracy for each real/fake level combination
    real_scores = score_df.query("label == 'real'")
    fake_scores = score_df.query("label == 'fake'")
    accuracy_df = []
    for real_level, real_df in real_scores.groupby("level", sort=False, dropna=False):
        for fake_level, fake_df in fake_scores.groupby("level", sort=False, dropna=False):
            acc = accuracy_score(
                y_true=[0] * len(real_df) + [1] * len(fake_df),
                y_pred=(pd.concat([real_df["score"], fake_df["score"]]).to_numpy() > 0.5).astype(int),
            )
            accuracy_df.append({"real_level": real_level, "fake_level": fake_level, "accuracy": acc})
    accuracy_df = pd.DataFrame(accuracy_df)

    # pivot the DataFrame to create a heatmap
    accuracy_df = accuracy_df.pivot(index="real_level", columns="fake_level", values="accuracy")
    accuracy_df.index = accuracy_df.index.to_series().fillna(110).values
    accuracy_df = accuracy_df.sort_index(axis=0, ascending=False)
    accuracy_df.columns = accuracy_df.columns.to_series().fillna(110).values
    accuracy_df = accuracy_df.sort_index(axis=1, ascending=False)

    # plot heatmap
    ax = sns.heatmap(
        accuracy_df,
        cmap="Blues",
        annot=False,
        fmt=".2f",
        annot_kws={"size": 6},
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        square=True,
        cbar=cbar,
        cbar_kws={"fraction": 0.0495, "pad": 0.02, "shrink": 0.98},
    )

    if cbar:
        colorbar = ax.collections[0].colorbar
        colorbar.set_ticks([0, 0.5, 1])
        labels = colorbar.ax.get_yticklabels()
        labels[0].set_verticalalignment("bottom")
        labels[-1].set_verticalalignment("top")

    ax.yaxis.set_inverted(False)
    ticks = np.array(range(len(score_df["level"].unique()))) + 0.5
    ax.set_xticks(ticks=ticks[::2], labels=score_df["level_label"].unique()[::2])
    ax.set_yticks(ticks=ticks[::2], labels=score_df["level_label"].unique()[::2])
    ax.tick_params("x", rotation=0)
    ax.tick_params("y", rotation=0)

    # configure plot
    plt.xlabel("JPEG Qual. (Fake)")
    plt.ylabel("JPEG Qual. (Real)")

    return ax.get_figure(), ax
