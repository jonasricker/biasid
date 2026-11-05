from collections.abc import Callable

from torch import Tensor, nn

from biasid.transforms import IdentityTransform


class DetectorWrapper(nn.Module):
    """
    Wrapper class for detectors.
    Preprocessing is applied to the image before call to 'forward', but after any transformation.
    Postprocessing is applied to the model's outputs.
    """

    def __init__(
        self,
        model: nn.Module,
        preprocessing: Callable | None = None,
        postprocessing: Callable | None = None,
    ) -> None:
        super().__init__()

        self.model = model
        self.preprocessing = preprocessing if preprocessing is not None else IdentityTransform()
        self.postprocessing = postprocessing if postprocessing is not None else IdentityTransform()

        self.device = next(self.model.parameters()).device
        self.model.eval()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    @property
    def id(self) -> str:
        return self.model.__class__.__name__.lower()
