# `biasid` - A Tool for Analyzing Transformation Bias and Robustness of AI-Generated Image Detectors

## Installation
```
pip install biasid
```

## Usage
```python
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision.transforms import v2

from biasid import DetectorWrapper, compute_metrics, create_bias_plot, get_score_df
from biasid.datasets import FolderDataset
from biasid.transforms import JPEG


# toy detector, use your own instead
class MyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(3 * 32 * 32, 1)

    def forward(self, images):
        return self.layer(images.flatten(start_dim=1))


if __name__ == "__main__":
    my_detector = MyDetector()

    # wrap your model into DetectorWrapper
    detector = DetectorWrapper(
        model=my_detector,
        preprocessing=v2.CenterCrop(32),
        postprocessing=torch.sigmoid,
    )

    # define datasets
    real = FolderDataset(root="real")
    fake = FolderDataset(root="fake")

    # define a transformation
    jpeg = JPEG()

    # compute scores
    score_df = get_score_df(
        detector=detector,
        real_dataset=real,
        fake_dataset=fake,
        output_dir="output/scores",
        transform=jpeg,
        batch_size=8,
        num_workers=4,
    )

    # compute metrics (with bootstrapping)
    metrics = compute_metrics(score_df)
    metrics.save("output/metrics")

    # visualize results
    create_bias_plot(result=metrics, real_subset="real", fake_subset="fake")
    plt.savefig("output/plots/jpeg.png")
```
