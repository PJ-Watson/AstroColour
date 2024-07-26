"""Various utility functions."""

import numpy as np
import numpy.typing as npt
from astropy.visualization import (
    AsinhStretch,
    BaseStretch,
    BaseTransform,
    LinearStretch,
    LogStretch,
    SqrtStretch,
)
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


def interpolate_colours(
    col_limits: npt.ArrayLike,
    N: int,
    hsv: bool = False,
    wavelengths: npt.ArrayLike | None = None,
    spacing: str = "linear",
):
    """
    _summary_.

    Parameters
    ----------
    col_limits : npt.ArrayLike
        _description_.
    N : int
        _description_.
    hsv : bool, optional
        _description_, by default False.
    wavelengths : npt.ArrayLike | None, optional
        _description_, by default None.
    spacing : str, optional
        _description_, by default "linear".

    Returns
    -------
    _type_
        _description_.

    Raises
    ------
    Exception
        _description_.
    """

    col_min = convert_colour_to_rgba(colour=col_limits[0], hsv=hsv)
    col_max = convert_colour_to_rgba(colour=col_limits[1], hsv=hsv)

    if N == 2:
        return np.row_stack([col_min, col_max])

    min_hsv = rgb_to_hsv(col_min[:-1])
    max_hsv = rgb_to_hsv(col_max[:-1])

    # hsv_diff
    if wavelengths is None:
        intervals = np.linspace(0, 1, N)
    else:
        wavelengths = np.asarray(wavelengths)
        assert (
            wavelengths.shape[0] == N
        ), "A central wavelength must be provided for each filter."
        intervals = (wavelengths - wavelengths[0]) / (wavelengths[-1] - wavelengths[0])
    match spacing:
        case "linear":
            stretch = LinearStretch()
        case "sqrt":
            stretch = SqrtStretch()
        case "log":
            stretch = LogStretch()
        case BaseStretch():
            stretch = spacing
        case _:
            raise Exception("`spacing` is not a valid parameter.")
    scaled_intervals = stretch(intervals)

    interp_hsv = (np.outer(scaled_intervals, (max_hsv - min_hsv))) + min_hsv

    interp_rgb = hsv_to_rgb(interp_hsv)
    interp_rgba = np.column_stack(
        (interp_rgb, np.linspace(col_min[-1], col_max[-1], N))
    )
    return interp_rgba


def circle_coords(centre, radius, image_shape):
    """
    Adapted from `skimage.draw`.

    Parameters
    ----------
    centre : _type_
        _description_.
    radius : _type_
        _description_.
    image_shape : _type_
        _description_.

    Returns
    -------
    _type_
        _description_.
    """

    centre = np.asarray(centre)

    upper_left = np.ceil(centre - radius).astype(int)
    lower_right = np.floor(centre + radius).astype(int)

    upper_left = np.maximum(upper_left, np.array([0, 0]))
    lower_right = np.minimum(lower_right, np.array(image_shape[:2]) - 1)

    shifted_centre = centre - upper_left
    bounding_shape = lower_right - upper_left + 1

    r_lim, c_lim = np.ogrid[0 : float(bounding_shape[0]), 0 : float(bounding_shape[1])]
    r, c = (r_lim - shifted_centre[0]), (c_lim - shifted_centre[1])
    rr, cc = np.nonzero((r**2 + c**2) < radius**2)
    rr.flags.writeable = True
    cc.flags.writeable = True
    rr += upper_left[0]
    cc += upper_left[1]
    return rr, cc


class RawAsinhTransform(BaseTransform):
    """
    A wrapper around AsinhStretch, where a is in non-normalised units.

    Parameters
    ----------
    transform_interval : :class:`astropy.visualization.BaseInterval`
        The interval transformation to apply to the image.
    a : float, optional
        The parameter a in the formula for AsinhStretch. This is given in raw
        data units rather than normalised to the range 0-1, by default 0.01.
    """

    def __init__(self, transform_interval, a=0.01):
        super().__init__()
        self.a = a
        self.transform_interval = transform_interval
        # self._transf_a()
        self.a_transf = a
        self.transform_asinh = AsinhStretch(a=self.a_transf)

    def __call__(self, values, clip=True):
        normed_values = self.transform_interval(values, clip=clip)
        vmin, vmax = self.transform_interval.get_limits(values)
        self.a_transf = np.subtract(self.a, float(vmin))

        if (vmax - vmin) != 0:
            self.a_transf = np.true_divide(self.a_transf, vmax - vmin)

        if clip:
            np.clip(self.a_transf, 0.0, 1.0, out=self.a_transf)

        self.transform_asinh = AsinhStretch(a=self.a_transf)

        return self.transform_asinh(
            self.transform_interval(values, clip=clip), clip=clip
        )

    @property
    def inverse(self):
        """
        The inverse transformation.

        This may not be accurate if the forward transformation has not been called
        already, as `a_transf` can only be fully defined relative to the raw data.

        Returns
        -------
        _type_
            _description_.
        """
        return self.__class__(
            self.transform_asinh.inverse, self.transform_interval.inverse
        )
