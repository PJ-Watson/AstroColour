"""Various utility functions."""

import numpy as np
from matplotlib.colors import ListedColormap, hsv_to_rgb, rgb_to_hsv, to_rgba


def gen_cmap_from_rgba(rgb, quantisation=2**16, invert=False):
    """
    _summary_.

    Parameters
    ----------
    rgb : _type_
        _description_.
    quantisation : _type_, optional
        _description_, by default 2**16.
    invert : bool, optional
        _description_, by default False.

    Returns
    -------
    _type_
        _description_.

    Raises
    ------
    Exception
        _description_.
    """

    if any(np.asarray(rgb) > 1):
        rgb = [r / 255 for r in rgb]

    if len(rgb) == 3:
        cmap = ListedColormap(
            np.c_[
                np.linspace(0, rgb[0], quantisation),
                np.linspace(0, rgb[1], quantisation),
                np.linspace(0, rgb[2], quantisation),
            ]
        )
    elif len(rgb) == 4:
        cmap = ListedColormap(
            np.c_[
                np.linspace(0, rgb[0], quantisation),
                np.linspace(0, rgb[1], quantisation),
                np.linspace(0, rgb[2], quantisation),
                np.linspace(0 if rgb[3] != 1 else 1.0, rgb[3], quantisation),
            ]
        )
    else:
        raise Exception("Invalid RGB(A) colour.")

    if invert:
        cmap = cmap.reversed()
    return cmap


def convert_colour_to_rgba(colour, hsv=False):
    """
    Convert a single colour to RGBA.

    Parameters
    ----------
    colour : _type_
        _description_.
    hsv : bool, optional
        _description_, by default False.

    Returns
    -------
    _type_
        _description_.
    """
    if hsv and type(colour) != str:
        if colour[0] > 1:
            _col = hsv_to_rgb((colour[0] / 360, colour[1], colour[2]))
        else:
            _col = hsv_to_rgb(colour)

        if len(colour) == 4:
            _col += (colour[3],)
            return _col
        else:
            _col += (1,)
            return _col
    else:
        return to_rgba(colour)


def interpolate_colours(col_limits, N, hsv=False):
    """
    _summary_.

    Parameters
    ----------
    col_limits : _type_
        _description_.
    N : _type_
        _description_.
    hsv : bool, optional
        _description_, by default False.

    Returns
    -------
    _type_
        _description_.
    """

    col_min = convert_colour_to_rgba(colour=col_limits[0], hsv=hsv)
    col_max = convert_colour_to_rgba(colour=col_limits[1], hsv=hsv)

    min_hsv = rgb_to_hsv(col_min[:-1])
    max_hsv = rgb_to_hsv(col_max[:-1])

    interp_hsv = np.linspace(min_hsv, max_hsv, N)
    interp_rgb = hsv_to_rgb(interp_hsv)
    interp_rgba = np.column_stack(
        (interp_rgb, np.linspace(col_min[-1], col_max[-1], N))
    )
    return interp_rgba
