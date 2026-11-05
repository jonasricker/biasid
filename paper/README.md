# Instructions for Reproducing the Results From the Paper

## Setup Environment
1. Install [uv](https://docs.astral.sh/uv) for your platform.
2. Run `uv sync` from within this directory.

## Download Data

### RAISE-1k (Original)
1. Download `RAISE_1k.csv` from [here](https://loki.disi.unitn.it/RAISE/confirm.php?package=1k) into this directory.
2. Run `scripts/download_raise1k.py`.
3. Copy `r0bf7f938t.png` from `missing_images` into `data/raise1k` to fix the corrupted .TIFF file.

### RAISE-1k (Resized)
1. Download `real_RAISE_1k.zip` from [here](https://www.grip.unina.it/download/prog/B-Free/extended_synthbuster/).
2. Extract the zip file and place the images into `data/raise1k_resized`.

### Synthbuster (with Extension)
1. Download the original Synthbuster dataset from [here](https://zenodo.org/records/10066460).
2. Delete the `firefly` subset.
3. Download `sd3_flux.zip` from [here](https://www.grip.unina.it/download/prog/B-Free/extended_synthbuster/).
4. Extract both zip files, merge them and place all subdirectories in `data/synthbuster`.

### SynthCLIC
SynthCLIC will be automatically downloaded and placed into HuggingFace's cache directory.

---

When all steps are completed, the `data` directory should look this:
```
├── raise1k
├── raise1k_resized
├── synthbuster
    ├── dalle2
    ├── dalle3
    ├── flux
    ├── glide
    ├── midjourney-v5
    ├── sd3_large
    ├── stable-diffusion-1-3
    ├── stable-diffusion-1-4
    ├── stable-diffusion-2
    └── stable-diffusion-xl
```

## Download Weights

The weights for UnivFD and RINE are already included in their respective code repositories in `external`.

### AIDE
Download the checkpoints from [here](https://drive.google.com/drive/folders/1qx76UFvDpgCxaPLBCmsA2WY-SSzeJrd4) and place them into `weights/aide`.

### B-Free
Download the checkpoints from [here](https://www.grip.unina.it/download/prog/B-Free/weights/BFREE_dino2reg4.zip) and place the directory `BFREE_dino2reg4` into `weights/bfree`.

### DRCT
Download the checkpoints from [here](https://modelscope.cn/datasets/BokingChen/DRCT-2M/resolve/master/pretrained.zip) and place the directories `DRCT-2M` and `GenImage` into `weights/drct`.

### SPAI
Download the checkpoint from [here](https://drive.google.com/file/d/1vvXmZqs6TVJdj8iF1oJ4L_fcgdQrp_YI/view) and place it into `weights/spai`.

---

When all steps are completed, the `weights` directory should look like this:
```
├── aide
│   ├── GenImage_train-001.pth
│   ├── progan_train-003.pth
│   └── sd14_train-002.pth
├── bfree
│   └── BFREE_dino2reg4
│       ├── config.yaml
│       └── model_epoch_best.pth
├── drct
│   ├── DRCT-2M
│   │   ├── sdv14
│   │   │   ├── clip-ViT-L-14_224_drct_amp_crop
│   │   │   │   └── 13_acc0.9664.pth
│   │   │   └── convnext_base_in22k_224_drct_amp_crop
│   │   │       └── 14_acc0.9996.pth
│   │   └── sdv2
│   │       ├── clip-ViT-L-14_224_drct_amp_crop
│   │       │   └── last_acc0.9112.pth
│   │       └── convnext_base_in22k_224_drct_amp_crop
│   │           └── 16_acc0.9993.pth
│   └── GenImage
│       └── sdv14
│           ├── clip-ViT-L-14_224_drct_amp_crop
│           │   └── 2_acc0.9558.pth
│           └── convnext_base_in22k_224_drct_amp_crop
│               └── last_acc0.9991.pth
└── spai
    └── spai.pth
```

## Replicate Our Experiments
You can reproduce all of our results by running
```bash
uv run replicate_results.py
```
Note that recomputing all results can take a considerable amount of time. However, results are cached between runs. All outputs will be written into `results`.

## External Code and Licenses
For all evaluated detectors we include the original code (with slight modifications for fixing imports), which may be subject to a different license:

| Detector | Venue     | Paper                | Repository          |
| -------- | --------- | -------------------- | ------------------- |
| UnivFD   | CVPR 2023 | [Link][univfd_paper] | [Link][univfd_repo] |
| DRCT     | ICML 2024 | [Link][drct_paper]   | [Link][drct_repo]   |
| RINE     | ECCV 2024 | [Link][rine_paper]   | [Link][rine_repo]   |
| AIDE     | ICLR 2025 | [Link][aide_paper]   | [Link][aide_repo]   |
| SPAI     | CVPR 2025 | [Link][spai_paper]   | [Link][spai_repo]   |
| B-Free   | CVPR 2025 | [Link][bfree_paper]  | [Link][bfree_repo]  |

[drct_paper]: https://proceedings.mlr.press/v235/chen24ay.html
[drct_repo]: https://github.com/beibuwandeluori/DRCT
[univfd_paper]: https://openaccess.thecvf.com/content/CVPR2023/html/Ojha_Towards_Universal_Fake_Image_Detectors_That_Generalize_Across_Generative_Models_CVPR_2023_paper.html
[univfd_repo]: https://github.com/WisconsinAIVision/UniversalFakeDetect
[bfree_paper]: https://openaccess.thecvf.com/content/CVPR2025/html/Guillaro_A_Bias-Free_Training_Paradigm_for_More_General_AI-generated_Image_Detection_CVPR_2025_paper.html
[bfree_repo]: https://github.com/grip-unina/B-Free
[aide_paper]: https://openreview.net/forum?id=ODRHZrkOQM
[aide_repo]: https://github.com/shilinyan99/AIDE
[rine_paper]: https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/9160_ECCV_2024_paper.php
[rine_repo]: https://github.com/mever-team/rine
[spai_paper]: https://openaccess.thecvf.com/content/CVPR2025/html/Karageorgiou_Any-Resolution_AI-Generated_Image_Detection_by_Spectral_Learning_CVPR_2025_paper.html
[spai_repo]: https://github.com/mever-team/spai
