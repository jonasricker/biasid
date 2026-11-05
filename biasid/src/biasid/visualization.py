import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import scienceplots  # noqa
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from biasid.metrics import METRIC_LABELS, MetricsResult
from biasid.transforms import TRANSFORM_UNITS


def create_bias_plot(
    result: MetricsResult, real_subset: str, fake_subset: str, title: str | None = None, clf_metric: str = "auroc"
) -> tuple[Figure, Axes]:
    """
    Visualize mean score, mean score shift and a classification metric, in one plot, with CIs.
    """
    real_query = "label == 'real' and subset == @real_subset"
    fake_query = "label == 'fake' and subset == @fake_subset"
    paired_query = "real_subset == @real_subset and fake_subset == @fake_subset"

    real_mean_score_df = result.mean_score.query(real_query)
    fake_mean_score_df = result.mean_score.query(fake_query)
    real_mss_df = result.mss.query(real_query)
    fake_mss_df = result.mss.query(fake_query)
    clf_df = result.clf.query(paired_query)
    transform = real_mean_score_df.query("transform != 'clean'")["transform"].iloc[0]

    _configure_mpl()
    _set_figsize(factor=(0.19, 0.32))
    fig, axs = plt.subplots(nrows=3, sharex=True, height_ratios=[1, 1, 0.75])
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

    _plot_lines(first=clf_df, second=None, metric=clf_metric, ax=axs[2])
    axs[2].set_ylim(-0.11, 1.11)
    axs[2].set_ylabel(METRIC_LABELS[clf_metric])
    axs[2].yaxis.set_major_locator(mticker.FixedLocator([0, 0.5, 1]))
    axs[2].yaxis.set_major_formatter(mticker.FixedFormatter(["0", "", "1"]))

    axs[2].set_xlabel(TRANSFORM_UNITS.get(transform.split("_")[0], transform))

    # only show every second label to avoid clutter
    xlabels = axs[2].get_xticklabels()
    if len(xlabels) > 6:
        for index, label in enumerate(xlabels):
            label.set_visible(index % 2 == 0)

    if len(real_mean_score_df) < 4:
        axs[2].set_xlim(axs[2].get_xlim()[0] - 0.1, axs[2].get_xlim()[1] + 0.1)

    fig.suptitle(title, y=1.075, x=0.572)

    return fig, axs


def _plot_lines(first: pd.DataFrame, second: pd.DataFrame | None, metric: str, ax: Axes | None) -> Axes:
    if ax is None:
        _, ax = plt.subplots()

    _plot_line_with_bands(
        xlabels=first["level_label"],  # type: ignore
        y=first[metric],
        lower=first[f"{metric}_ci_lower"],
        upper=first[f"{metric}_ci_upper"],
        label="Real" if second is not None else None,
        ax=ax,
    )

    if second is not None:
        _plot_line_with_bands(
            xlabels=second["level_label"],  # type: ignore
            y=second[metric],
            lower=second[f"{metric}_ci_lower"],
            upper=second[f"{metric}_ci_upper"],
            label="Fake",
            ax=ax,
        )

    return ax


def _plot_line_with_bands(
    xlabels: list[str], y: ArrayLike, lower: ArrayLike, upper: ArrayLike, label: str | None, ax: Axes
) -> Axes:
    if ax is None:
        _, ax = plt.subplots()

    if label == "Real":
        marker = "o"
        color = "#CC6677"
        fillstyle = "none"
    elif label == "Fake":
        marker = "x"
        color = "#332288"
        fillstyle = "full"
    else:
        marker = "."
        color = "#117733"
        fillstyle = "full"

    xticks = range(len(xlabels))
    (line,) = ax.plot(xticks, y, marker=marker, label=label, color=color, fillstyle=fillstyle)
    ax.fill_between(xticks, lower, upper, color=line.get_color(), alpha=0.33, linewidth=0)

    ax.set_xticks(ticks=xticks, labels=xlabels)

    return ax


def _configure_mpl():
    plt.style.use(["science", "no-latex", "grid", "muted"])

    plt.rcParams.update(
        {
            "figure.constrained_layout.use": True,
            "figure.constrained_layout.h_pad": -0.01,
            "figure.dpi": 300,
            "figure.titlesize": 7,
            "axes.labelsize": 6,
            "axes.linewidth": 0.33,
            "font.size": 8,
            "legend.fontsize": 5,
            "legend.handlelength": 1.5,
            "legend.columnspacing": 1.0,
            "legend.handletextpad": 0.5,
            "xtick.labelsize": 5,
            "ytick.labelsize": 5,
            "axes.labelpad": -0.2,
            "xtick.major.pad": 1.0,
            "ytick.major.pad": 1.0,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "xtick.minor.size": 0,
            "ytick.minor.size": 0,
            "lines.linewidth": 0.5,
            "lines.markersize": 3,
            "lines.markeredgewidth": 0.5,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.01,
            "grid.alpha": 0.33,
            "grid.linewidth": 0.33,
            "path.simplify": False,
        }
    )


def _set_figsize(
    page_width: float = 397.48499,
    ratio: float = 2 / (1 + 5**0.5),
    factor: float | tuple[float, float] = 1.0,
    nrows: int = 1,
    ncols: int = 1,
) -> None:
    height = page_width * ratio * (nrows / ncols)
    if not isinstance(factor, tuple):
        factor = (factor, factor)
    plt.rcParams["figure.figsize"] = page_width * 100 / 7227 * factor[0], height * 100 / 7227 * factor[1]
