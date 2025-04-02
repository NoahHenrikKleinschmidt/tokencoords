"""
Coordinate to tokens converter for Paligemma.
"""

import numpy as np
import re

from .converter import CoordsTokensConverter

NTOKENS = 1024
"""
The number of <locX> tokens used by Paligemma.
"""

token_pattern = re.compile(r"<loc(\d{4})>")


def token_formatter(index):
    """
    Create a Paligemma <locX> token for a given index.
    """
    return f"<loc{index:0>4}>"


class PaligemmaConverter(CoordsTokensConverter):
    """
    Coordinate to tokens converter for use with Paligemma.
    This class can produce <loc0000> to <loc1023> location tokens
    for paired y,x (pixel in height, pixel in width, starting from topleft) coordinates.

    Parameters
    ----------
    image_size : tuple
        The size of the image in pixels (height, width).
    """

    def __init__(self, image_size: tuple):
        super().__init__(image_size, NTOKENS, token_formatter)

    def coord_to_token(self, coord: np.ndarray) -> int:
        y, x = coord / self.image_size
        y = max(0, int(y * self.ntokens) - 1)
        x = max(0, int(x * self.ntokens) - 1)

        ytoken = self.make_token_for_index(y)
        xtoken = self.make_token_for_index(x)

        return ytoken + xtoken

    def token_to_coord(self, token: str) -> np.ndarray:
        locs = []
        for loc in token_pattern.findall(token):
            loc = int(loc)
            if loc >= self.ntokens:
                raise ValueError(
                    f"Token {token} is out of bounds for ntokens {self.ntokens}"
                )
            loc = loc / self.ntokens
            locs.append(loc)

        locs = np.array(locs)
        locs = locs * self.image_size
        locs = locs.squeeze()
        return locs


def points_to_tokens(points: np.ndarray, image_shape: tuple) -> np.ndarray:
    """
    Convert one or more points to tokens.

    Parameters
    ----------
    points : np.ndarray
        The points to convert.
        Should be of shape (N, 2) where N is the number of points and
        each point is represented by (y, x).
    image_shape : tuple
        The shape of the image (height, width).

    Returns
    -------
    np.ndarray
        The tokenized points.
    """
    if not isinstance(points, np.ndarray):
        points = np.array(points)

    if points.ndim == 1 and points.shape[0] == 2:
        points = points.reshape(1, -1)

    if points.shape[1] != 2:
        raise ValueError(f"Expected points to have shape (N, 2), got {points.shape}")

    converter = PaligemmaConverter(image_shape)
    tokens = converter.forward(points)
    return tokens


def tokens_to_points(tokens: np.ndarray, image_shape: tuple) -> np.ndarray:
    """
    Convert one or more tokenized points back to coordinates.

    Parameters
    ----------
    tokens : np.ndarray
        The tokenized points to convert.
        Should be of shape (N,) where N is the number of points and
        each point is represented by a string containing 2 <locX> tokens.
    image_shape : tuple
        The shape of the image (height, width).

    Returns
    -------
    np.ndarray
        The points coordinates.
    """
    if not isinstance(tokens, (list, tuple, np.ndarray)):
        tokens = np.array(tokens)

    converter = PaligemmaConverter(image_shape)
    coords = converter.reverse(tokens)
    return coords


def bbox_to_tokens(bbox_coords: np.ndarray, image_shape: tuple) -> np.ndarray:
    """
    Convert one or more bounding boxes to tokens.

    Parameters
    ----------
    bbox_coords : np.ndarray
        The bounding box coordinates to convert.
        Should be of shape (N, 4) where N is the number of boxes and
        each box is represented by (ymin, xmin, ymax, xmax).
    image_shape : tuple
        The shape of the image (height, width).

    Returns
    -------
    np.ndarray
        The tokenized bounding box coordinates.
    """
    if not isinstance(bbox_coords, np.ndarray):
        bbox_coords = np.array(bbox_coords)

    if bbox_coords.ndim == 1 and bbox_coords.shape[0] == 4:
        bbox_coords = bbox_coords.reshape(1, -1)

    if bbox_coords.shape[1] != 4:
        raise ValueError(
            f"Expected bbox_coords to have shape (N, 4), got {bbox_coords.shape}"
        )

    starts = bbox_coords[:, :2]
    ends = bbox_coords[:, 2:]

    converter = PaligemmaConverter(image_shape)
    start_tokens = converter.forward(starts)
    end_tokens = converter.forward(ends)

    tokens = np.stack((start_tokens, end_tokens), axis=1)
    tokens = map(
        lambda x: x[0] + x[1],
        tokens,
    )
    tokens = np.array(list(tokens))
    return tokens


_length_of_2_loc_tokens = 18


def tokens_to_bbox(tokens: np.ndarray, image_shape: tuple) -> np.ndarray:
    """
    Convert one or more tokenized bounding boxes back to coordinates.

    Parameters
    ----------
    tokens : np.ndarray
        The tokenized bounding box coordinates to convert.
        Should be of shape (N,) where N is the number of boxes and
        each box is represented by a string containing 4 <locX> tokens.
    image_shape : tuple
        The shape of the image (height, width).

    Returns
    -------
    np.ndarray
        The bounding box coordinates.
    """
    if not isinstance(tokens, (list, tuple, np.ndarray)):
        tokens = np.array(tokens)

    tokens = (i.strip() for i in tokens)
    tokens = [
        [i[:_length_of_2_loc_tokens], i[_length_of_2_loc_tokens:]] for i in tokens
    ]
    tokens = np.array(tokens)
    start_tokens = tokens[:, 0]
    end_tokens = tokens[:, 1]

    converter = PaligemmaConverter(image_shape)
    start_coords = converter.reverse(start_tokens)
    end_coords = converter.reverse(end_tokens)

    bboxes = np.concatenate((start_coords, end_coords), axis=1)
    return bboxes


paligemma_converter_224 = PaligemmaConverter((224, 224))
paligemma_converter_448 = PaligemmaConverter((448, 448))
paligemma_converter_896 = PaligemmaConverter((896, 896))
