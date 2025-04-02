import numpy as np
from pytest import approx

image_shape = (224, 224)


def test_convert_single():

    import tokencoords.paligemma as ict

    conv = ict.paligemma_converter_224

    point = np.array([0, 100])
    token = conv.forward(point)
    rev = conv.reverse(token)
    rev = rev.squeeze()
    assert rev == approx(point, rel=1e-1)


def test_convert_multiple():
    import tokencoords.paligemma as ict

    conv = ict.paligemma_converter_224

    points = np.array([[0, 100], [10, 20]])
    tokens = conv.forward(points)
    rev = conv.reverse(tokens)
    assert rev == approx(points, rel=1e-1)


def test_convert_bboxes():
    import tokencoords.paligemma as ict

    bboxes = np.array(
        [
            [5, 5, 10, 20],
            [10, 20, 30, 40],
        ]
    )
    tokens = ict.bbox_to_tokens(bboxes, image_shape)
    rev = ict.tokens_to_bbox(tokens, image_shape)
    rev = rev.squeeze()
    assert rev == approx(bboxes, rel=1e-1)
