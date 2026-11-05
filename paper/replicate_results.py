from collections import defaultdict
from pathlib import Path

import lightning as L
import pandas as pd
import torch
from biasid import DetectorWrapper, compute_metrics, create_bias_plot, get_score_df
from biasid.datasets import FolderDataset, LabelDataset, SynthCLIC, TransformDataset
from biasid.metrics import MetricsResult
from biasid.transforms import JPEG, Downscaling, Grayscale, Rotation, Upscaling, WebP
from lightning.pytorch.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchvision.transforms import v2

from detectors.aide import AIDE
from detectors.bfree import BFree
from detectors.drct import DRCT
from detectors.rine import RINE
from detectors.simple_detector import SimpleDetector
from detectors.spai import SPAI
from detectors.univfd import UnivFD
from utils import (
    DETECTOR_LABELS,
    create_ats_row,
    create_plot_value_df,
    format_float,
    plot_accuracy_heatmap,
    plot_score_and_mss,
    render_bias_analysis_table,
)

RESULT_DIR = Path("new_results")
BIAS_ANALYSIS_DIR = RESULT_DIR / "bias_analysis"
SAMPLE_SIZE_ANALYSIS_DIR = RESULT_DIR / "sample_size_analysis"
ILLUSTRATIVE_EXAMPLE_DIR = RESULT_DIR / "illustrative_example"


def bias_analysis():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DETECTORS = [
        (UnivFD, "external/univfd/pretrained_weights/fc_weights.pth"),
        (
            DRCT,
            "weights/drct/DRCT-2M/sdv2/clip-ViT-L-14_224_drct_amp_crop/last_acc0.9112.pth",
        ),  # best in Tab. 3
        (RINE, "external/rine/ckpt/model_ldm_trainable.pth"),  # best in Tab. 6
        (AIDE, "weights/aide/GenImage_train-001.pth"),
        (SPAI, "weights/spai/spai.pth"),
        (BFree, "weights/bfree"),
    ]

    DATASETS = [
        (
            FolderDataset(root="data/raise1k_resized"),
            FolderDataset(root="data/synthbuster"),
        ),
        (SynthCLIC(label="real", split="all"), SynthCLIC(label="fake", split="all")),
        (
            FolderDataset(root="data/raise1k"),
            FolderDataset(root="data/synthbuster"),
        ),
    ]

    TRANSFORMS = [
        JPEG(),
        WebP(),
        Upscaling(mode="bilinear"),
        Downscaling(mode="bilinear"),
        Rotation(),
        Grayscale(),
        Upscaling(mode="bicubic"),
        Downscaling(mode="bicubic"),
    ]

    plot_value_dfs = []
    ats_rows = []

    for detector_cls, checkpoint_path in DETECTORS:
        model = detector_cls(checkpoint_path=checkpoint_path).to(device)
        detector = DetectorWrapper(
            model=model,
            preprocessing=model.preprocessing,
            postprocessing=model.postprocessing,
        )
        for real_dataset, fake_dataset in DATASETS:
            for transform in TRANSFORMS:
                # only perform analysis of original-size RAISE-1k for selected transforms
                if real_dataset.id == "raise1k" and transform.id not in [
                    "jpeg",
                    "upscaling_bilinear",
                    "downscaling_bilinear",
                ]:
                    continue

                score_df = get_score_df(
                    detector=detector,
                    real_dataset=real_dataset,
                    fake_dataset=fake_dataset,
                    transform=transform,
                    output_dir=BIAS_ANALYSIS_DIR / "scores" / detector.id,
                    batch_size=16 if detector.id != "spai" else 1,
                    num_workers=8 if detector.id != "spai" else 1,
                )

                out_dir = (
                    BIAS_ANALYSIS_DIR
                    / "per_detector_results"
                    / detector.id
                    / f"{real_dataset.id}_vs_{fake_dataset.id}"
                    / transform.id
                )
                out_dir.mkdir(parents=True, exist_ok=True)

                # compute and save metrics
                if (out_dir / "mean_score.csv").exists():
                    metrics = MetricsResult.load(out_dir, index_col=False)
                else:
                    metrics = compute_metrics(score_df=score_df)
                    metrics.save(out_dir, index=False)

                # save detailed values of plots
                plot_value_dfs.append(
                    create_plot_value_df(
                        metrics,
                        real_dataset_id=real_dataset.id,
                        fake_dataset_id=fake_dataset.id,
                        detector_id=detector.id,
                    )
                )

                # create bias plot for each real/fake subset combination
                for real_subset in metrics.mean_score.query("label == 'real'")["subset"].unique():
                    for fake_subset in metrics.mean_score.query("label == 'fake'")["subset"].unique():
                        fig, _ = create_bias_plot(
                            result=metrics,
                            real_subset=real_subset,
                            fake_subset=fake_subset,
                            title=DETECTOR_LABELS[detector.id],
                        )
                        fig.savefig(out_dir / f"{real_subset}_vs_{fake_subset}.pdf")
                        plt.close()

                # extract data for ATS tables
                ats_rows.append(
                    create_ats_row(
                        metrics,
                        real_dataset_id=real_dataset.id,
                        fake_dataset_id=fake_dataset.id,
                        detector_id=detector.id,
                    )
                )

    # create tables (one per real/fake setting) with exact plot values for appendix
    out_dir = BIAS_ANALYSIS_DIR / "plot_tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    dfs_per_setting = defaultdict(list)
    for df in plot_value_dfs:
        setting = df.index.get_level_values("setting")[0]
        dfs_per_setting[setting].append(df.droplevel("setting"))

    for setting, dfs in dfs_per_setting.items():
        dfs_per_detector = defaultdict(list)
        for df in dfs:
            detector = df.columns.get_level_values("detector")[0]
            dfs_per_detector[detector].append(df)
        dfs_per_detector = {detector: pd.concat(dfs, axis=0) for detector, dfs in dfs_per_detector.items()}
        setting_df = pd.concat(dfs_per_detector.values(), axis=1)
        setting_df.to_latex(out_dir / f"{setting}.tex", float_format=format_float, multicolumn_format="c")

    # create ATS tables for main paper
    ats_df = pd.concat(ats_rows, ignore_index=True)
    out_dir = BIAS_ANALYSIS_DIR / "ats_tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    for transform, transform_df in ats_df.query("dataset != 'Synthbuster (original RAISE-1k)'").groupby(
        "transform", sort=False
    ):
        render_bias_analysis_table(transform_df, path=out_dir / f"synthbuster_synthclic_{transform}.tex")

    # create ATS tables for resized vs. original RAISE-1k comparison in appendix
    for transform, transform_df in ats_df.query("dataset != 'SynthCLIC'").groupby("transform", sort=False):
        if transform in [
            "jpeg",
            "upscaling_bilinear",
            "downscaling_bilinear",
        ]:
            render_bias_analysis_table(transform_df, path=out_dir / f"raise_resized_original_{transform}.tex")


def sample_size_analysis():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DETECTORS = [
        (UnivFD, "external/univfd/pretrained_weights/fc_weights.pth"),
        (
            DRCT,
            "weights/drct/DRCT-2M/sdv2/clip-ViT-L-14_224_drct_amp_crop/last_acc0.9112.pth",
        ),  # best in Tab. 3
        (RINE, "external/rine/ckpt/model_ldm_trainable.pth"),  # best in Tab. 6
        (AIDE, "weights/aide/GenImage_train-001.pth"),
        (SPAI, "weights/spai/spai.pth"),
        (BFree, "weights/bfree"),
    ]

    DATASETS = [
        (
            FolderDataset(root="data/raise1k_resized"),
            FolderDataset(root="data/synthbuster"),
        ),
    ]

    TRANSFORMS = [
        JPEG(),
    ]

    ats_rows = []

    for detector_cls, checkpoint_path in DETECTORS:
        model = detector_cls(checkpoint_path=checkpoint_path).to(device)
        detector = DetectorWrapper(
            model=model,
            preprocessing=model.preprocessing,
            postprocessing=model.postprocessing,
        )
        for real_dataset, fake_dataset in DATASETS:
            for transform in TRANSFORMS:
                score_df = get_score_df(
                    detector=detector,
                    real_dataset=real_dataset,
                    fake_dataset=fake_dataset,
                    transform=transform,
                    output_dir=BIAS_ANALYSIS_DIR / "scores" / detector.id,
                    batch_size=16 if detector.id != "spai" else 1,
                    num_workers=8 if detector.id != "spai" else 1,
                )

                for n_samples in [50, 100, 500, 1000]:
                    out_dir = (
                        SAMPLE_SIZE_ANALYSIS_DIR
                        / "per_detector_results"
                        / detector.id
                        / f"{real_dataset.id}_vs_{fake_dataset.id}"
                        / transform.id
                        / f"{n_samples}_samples"
                    )
                    out_dir.mkdir(parents=True, exist_ok=True)

                    # form random sample with consistent samples across subsets
                    image_ids = score_df["image_id"].drop_duplicates()

                    sampled_image_ids = image_ids.sample(n=n_samples, random_state=42).sort_values()

                    score_df_subset = score_df[score_df["image_id"].isin(sampled_image_ids)].copy()

                    # compute and save metrics
                    if (out_dir / "mean_score.csv").exists():
                        metrics = MetricsResult.load(out_dir, index_col=False)
                    else:
                        metrics = compute_metrics(score_df=score_df_subset)
                        metrics.save(out_dir, index=False)

                    # extract data for ATS tables
                    ats_rows.append(
                        create_ats_row(
                            metrics,
                            real_dataset_id=real_dataset.id,
                            fake_dataset_id=fake_dataset.id,
                            detector_id=detector.id,
                            n_samples=n_samples,
                            ci_gray=False,
                        )
                    )

    ats_df = pd.concat(ats_rows, ignore_index=True).drop(columns=["dataset", "transform", "Acc. Shift", "AUC Shift"])
    n_samples_col = ats_df.pop("n_samples")
    ats_df.insert(1, "n_samples", n_samples_col)
    ats_df = ats_df.set_index("detector")
    ats_df.to_latex(SAMPLE_SIZE_ANALYSIS_DIR / f"{real_dataset.id}_vs_{fake_dataset.id}_{transform.id}.tex")


def illustrative_example():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L.seed_everything(42, workers=True)
    generator = torch.Generator()
    generator.manual_seed(42)

    # set up datasets
    real_ds = FolderDataset(root="data/raise1k_resized")
    real_train, real_test = random_split(real_ds, lengths=[800, 200], generator=generator)
    real_train = LabelDataset(real_train, label=0)
    generator.manual_seed(42)  # seed again to get same split
    fake_ds = FolderDataset(root="data/synthbuster/flux")
    fake_train, fake_test = random_split(fake_ds, lengths=[800, 200])
    fake_train = LabelDataset(fake_train, label=1)

    def _train_and_test(setting: str) -> None:
        setting_dir = ILLUSTRATIVE_EXAMPLE_DIR / setting

        detector = SimpleDetector(backbone="clipvitl14")

        if not setting_dir.exists():
            torch.compile(detector)

            if setting == "a":
                random_jpeg = v2.Compose(
                    [
                        v2.ToDtype(dtype=torch.uint8, scale=True),
                        v2.JPEG(quality=(1, 100)),
                        v2.ToDtype(dtype=torch.float32, scale=True),
                    ]
                )

                real_train_tf = TransformDataset(
                    real_train, transform=v2.Compose([random_jpeg, v2.RandomCrop(224), detector.transform])
                )
                fake_train_tf = TransformDataset(
                    fake_train, transform=v2.Compose([random_jpeg, v2.RandomCrop(224), detector.transform])
                )
            elif setting == "b":
                fixed_jpeg = v2.Compose(
                    [
                        v2.ToDtype(dtype=torch.uint8, scale=True),
                        v2.JPEG(quality=80),
                        v2.ToDtype(dtype=torch.float32, scale=True),
                    ]
                )

                real_train_tf = TransformDataset(
                    real_train, transform=v2.Compose([fixed_jpeg, v2.RandomCrop(224), detector.transform])
                )
                fake_train_tf = TransformDataset(
                    fake_train, transform=v2.Compose([v2.RandomCrop(224), detector.transform])
                )
            elif setting == "c":
                fixed_jpeg = v2.Compose(
                    [
                        v2.ToDtype(dtype=torch.uint8, scale=True),
                        v2.JPEG(quality=80),
                        v2.ToDtype(dtype=torch.float32, scale=True),
                    ]
                )

                real_train_tf = TransformDataset(
                    real_train, transform=v2.Compose([v2.RandomCrop(224), detector.transform])
                )
                fake_train_tf = TransformDataset(
                    fake_train, transform=v2.Compose([fixed_jpeg, v2.RandomCrop(224), detector.transform])
                )
            else:
                raise ValueError

            train_ds = ConcatDataset([real_train_tf, fake_train_tf])
            train_dl = DataLoader(train_ds, batch_size=16, num_workers=8, shuffle=True)

            trainer = L.Trainer(
                devices=1,
                callbacks=[
                    ModelCheckpoint(save_top_k=0, save_last=True, dirpath=setting_dir),
                ],
                max_epochs=10,
            )

            trainer.fit(model=detector, train_dataloaders=train_dl)

        wrapper = DetectorWrapper(detector.to(device), preprocessing=detector.transform)

        score_df = get_score_df(
            detector=wrapper,
            real_dataset=real_test,
            fake_dataset=fake_test,
            transform=JPEG(),
            output_dir=setting_dir / "scores",
            batch_size=16,
            num_workers=8,
        )

        metrics = compute_metrics(score_df=score_df)
        metrics.save(setting_dir / "metrics", index=False)

        fig, _ = plot_score_and_mss(
            result=metrics,
            real_subset="raise1k_resized",
            fake_subset="flux",
        )
        fig.savefig(ILLUSTRATIVE_EXAMPLE_DIR / f"{setting}_raise1k_resized_vs_flux_jpeg_scores.pdf")
        plt.close()

        fig, _ = plot_accuracy_heatmap(score_df)
        fig.savefig(ILLUSTRATIVE_EXAMPLE_DIR / f"{setting}_raise1k_resized_vs_flux_jpeg_heatmap.pdf")
        plt.close()

    _train_and_test(setting="a")
    _train_and_test(setting="b")
    _train_and_test(setting="c")


if __name__ == "__main__":
    bias_analysis()
    sample_size_analysis()
    illustrative_example()
