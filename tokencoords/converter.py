"""
Turn coordinates within images into tokens.
"""

import numpy as np


class CoordsTokensConverter:
    """
    The base class to convert coordinates to a tokenized format for use with VLMs and vice versa.

    Parameters
    ----------
    image_size : tuple
        The size of the images that will be passed to the tokenizer.
        This should be a tuple of the form (height, width).
    ntokens : int
        The number of tokens to use for the coordinates.
        This should be the number of unique tokens that can be generated.
    token_formatter : callable
        A function that takes an index as first (and only required) argument and returns a string.
        The function must be able to handle additional arguments and keyword arguments as (args and kwargs) to avoid crashes.
        The function must return a string that will be used as the token.
    """

    def __init__(self, image_size: tuple, ntokens: int, token_formatter: callable):
        self.image_size = np.array(image_size)
        self.ntokens = ntokens
        self.token_formatter = token_formatter

    def forward(self, coords: np.ndarray) -> np.ndarray:
        """
        Convert coordinates to tokens.

        Parameters
        ----------
        coords : np.ndarray
            The coordinates to convert.

        Returns
        -------
        np.ndarray
            The tokenized coordinates.
        """
        if not isinstance(coords, np.ndarray):
            coords = np.array(coords)
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)

        tokens = [None] * len(coords)
        for i, coord in enumerate(coords):
            token = self._coord_to_token_call(coord)
            tokens[i] = token

        return np.array(tokens)

    def reverse(self, tokens: np.ndarray) -> np.ndarray:
        """
        Convert tokens back to coordinates.

        Parameters
        ----------
        tokens : np.ndarray
            The tokens to convert.

        Returns
        -------
        np.ndarray
            The coordinates.
        """
        if not isinstance(tokens, np.ndarray):
            tokens = np.array(tokens)
        if tokens.shape[0] == 1:
            tokens = tokens.reshape(1, -1)
        coords = [None] * len(tokens)
        for i, token in enumerate(tokens):
            coords[i] = self._token_to_coord_call(token)
        return np.array(coords)

    def coord_to_token(self, coord: np.ndarray) -> str:
        """
        Get a tokenized representation for a given coordinate.

        Parameters
        ----------
        coord : np.ndarray
            The coordinate to get the index for.

        Returns
        -------
        str
            The token string.
        """
        raise NotImplementedError("This method should be implemented in a subclass.")

    def token_to_coord(self, token: str) -> np.ndarray:
        """
        Get the coordinate for a given index.

        Parameters
        ----------
        token : str
            The token to get the coordinate for.

        Returns
        -------
        np.ndarray
            The coordinate.
        """
        raise NotImplementedError("This method should be implemented in a subclass.")

    def make_token_for_index(self, index: int) -> str:
        """
        Create a token for a given index.

        Parameters
        ----------
        index : int
            The index to create a token for.

        Returns
        -------
        str
            The token string.
        """
        if index < 0 or index >= self.ntokens:
            raise ValueError(
                f"Index {index} is out of bounds for ntokens {self.ntokens}."
            )
        return self.token_formatter(index)

    def _make_coord_tokens(self):
        """
        Create a list of tokens for the coordinates.
        """
        coord_tokens = [None] * self.ntokens
        for i in range(self.ntokens):
            coord_tokens[i] = self.make_token_for_index(i)
        return coord_tokens

    def _coord_to_token_call(self, coord):
        if coord.shape != (2,):
            raise ValueError(f"Expected coord to be of shape (2,), got {coord.shape}")
        return self.coord_to_token(coord)

    def _token_to_coord_call(self, token: str) -> np.ndarray:
        if isinstance(token, (np.ndarray, list)) and len(token) == 1:
            token = token[0]
        if not isinstance(token, str):
            raise ValueError(f"Expected token to be a string, got {type(token)}")
        return self.token_to_coord(token)

    def __call__(self, inputs: np.ndarray, *args, **kwargs):
        if not isinstance(inputs, np.ndarray):
            inputs = np.array(inputs)

        if np.issubdtype(inputs.dtype, np.number):
            return self.forward(inputs, *args, **kwargs)
        elif np.issubdtype(inputs.dtype, np.str_):
            return self.reverse(inputs)
        else:
            raise TypeError(
                "Input array must contain either numeric values or strings."
            )

    def __repr__(self):
        return f"{self.__class__.__name__}(image_size={self.image_size}, ntokens={self.ntokens})"
