from biasid.transforms import JPEG, Grayscale, Upscaling


def test_transform_name(transform):
    assert transform.id == "mocktransform"
    assert Upscaling().id == "upscaling_bilinear"


def test_transform_level_labels():
    assert JPEG().clean_level_label == "-"
    assert JPEG().get_level_label(90) == "90"
    assert Grayscale().clean_level_label == "RGB"
    assert Grayscale().get_level_label(1) == "Gray"
