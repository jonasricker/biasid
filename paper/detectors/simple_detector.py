import lightning as L
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torchmetrics.classification import BinaryAUROC
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.transforms import v2
from transformers import CLIPConfig, CLIPImageProcessor, CLIPVisionModel


class ResNet50Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.out_features = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self._transform = v2.Compose(
            [v2.CenterCrop(224), v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

        self.size = 224

    def transform(self, x: Tensor) -> Tensor:
        return self._transform(x)

    def forward(self, x: Tensor):
        return self.model(x)


class CLIPBackbone(nn.Module):
    def __init__(
        self, model_name: str = "openai/clip-vit-large-patch14", freeze: bool = True, do_resize: bool = False
    ) -> None:
        super().__init__()

        config = CLIPConfig.from_pretrained(model_name)
        self.model = CLIPVisionModel.from_pretrained(model_name, config=config.vision_config)
        self.do_resize = do_resize

        if freeze:
            self.model.vision_model.requires_grad_(False)

        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        self.out_features = self.model.config.hidden_size

        self.size = self.processor.size["shortest_edge"]

    def transform(self, x: Tensor) -> Tensor:
        processed = self.processor(
            images=x, return_tensors="pt", do_rescale=False, do_resize=self.do_resize, do_center_crop=True
        )
        return processed["pixel_values"][0]

    def forward(self, x: Tensor):
        return self.model(x)["pooler_output"]


class SimpleDetector(L.LightningModule):
    def __init__(self, backbone: str, lr: float = 0.001, backbone_kwargs: dict | None = None):
        super().__init__()
        self.save_hyperparameters()

        if backbone_kwargs is None:
            backbone_kwargs = {}
        if backbone == "resnet50":
            self.backbone = ResNet50Backbone(**backbone_kwargs)
        elif backbone == "clipvitl14":
            self.backbone = CLIPBackbone(**backbone_kwargs)
        else:
            raise NotImplementedError

        classifier = nn.Linear(in_features=self.backbone.out_features, out_features=1)
        self.model = nn.Sequential(self.backbone, classifier)

        # define metrics
        self.train_auc = BinaryAUROC()
        self.val_auc = BinaryAUROC()
        self.test_auc = BinaryAUROC()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x).sigmoid()

    def transform(self, x: Tensor) -> Tensor:
        return self.backbone.transform(x)

    def training_step(self, batch: Tensor) -> Tensor:
        x, y_true = batch
        logits = self.model(x).squeeze(1)
        y_pred = logits.sigmoid()
        loss = F.binary_cross_entropy_with_logits(input=logits, target=y_true.float())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_auc(y_pred, y_true)
        self.log("train_auc", self.train_auc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Tensor) -> None:
        x, y_true = batch
        logits = self.model(x).squeeze(1)
        y_pred = logits.sigmoid()
        loss = F.binary_cross_entropy_with_logits(input=logits, target=y_true.float())
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.val_auc(y_pred, y_true)
        self.log("val_auc", self.val_auc, on_step=True, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Tensor) -> None:
        x, y_true = batch
        logits = self.model(x).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(input=logits, target=y_true.float())
        y_pred = logits.sigmoid()
        self.log("test_loss", loss, on_step=True, on_epoch=True)
        self.test_auc(y_pred, y_true)
        self.log("test_auc", self.test_auc, on_step=True, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> Optimizer:
        optimizer = AdamW(params=[p for p in self.parameters() if p.requires_grad], lr=self.hparams.lr)
        return optimizer
