"""Primary module for the AstroColour package."""

from pathlib import Path

import astropy.io.fits as pf
import astropy.visualization as visualization
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from . import utils


class ColourImage:
    """
    Placeholder docstring.

    Parameters
    ----------
    images : npt.ArrayLike
        _description_.
    img_extns : npt.ArrayLike | None, optional
        _description_, by default None.
    crop : npt.ArrayLike | None, optional
        _description_, by default None.
    transformation : _type_, optional
        _description_, by default None.
    transformation_kwargs : _type_, optional
        _description_, by default None.
    colours : _type_, optional
        _description_, by default None.
    colours_kwargs : _type_, optional
        _description_, by default None.
    """

    def __init__(
        self,
        images: npt.ArrayLike,
        img_extns: npt.ArrayLike | None = None,
        crop: npt.ArrayLike | None = None,
        transformation=None,
        transformation_kwargs=None,
        colours=None,
        colours_kwargs=None,
    ):
        self._validate_images_input(images, img_extns)

        if transformation_kwargs is None:
            transformation_kwargs = {}
        self.apply_transformation(transformation, **transformation_kwargs)

        if colours_kwargs is None:
            colours_kwargs = {}
        self.generate_colours(colours, **colours_kwargs)

        self.apply_colours()
        self.combine_to_rgb()
        self.plot()

    def _validate_images_input(self, images, img_extns):

        if isinstance(images, str | Path):
            self._validate_imgs_path(images, img_extns)
        else:
            self._validate_imgs_array(images)

    def _validate_imgs_array(self, images):

        self.images = []
        for img in images:
            img_view = np.atleast_2d(img)
            if img_view.ndim == 2:
                self.images.append(
                    np.asarray(img).reshape(img_view.shape[0], img_view.shape[1])
                )
            if img_view.ndim > 2:
                self.images.append(
                    np.nanmedian(img_view, axis=0).reshape(
                        img_view.shape[1], img_view.shape[2]
                    )
                )

        self.image_shape = self.images[0].shape
        self.N = len(self.images)
        self.images = np.asarray(self.images)
        assert all(
            [i.shape == self.image_shape for i in self.images]
        ), "The dimensions of the input images do not match."

    def apply_transformation(
        self,
        transformation: visualization.BaseTransform | None = None,
        per_image: bool = False,
    ):
        """
        _summary_.

        Parameters
        ----------
        transformation : visualization.BaseTransform | None, optional
            _description_, by default None.
        per_image : bool, optional
            _description_, by default False.
        """

        if transformation is not None:
            assert isinstance(
                transformation, visualization.BaseTransform
            ), "`transformation` must be an instance of `astropy.visualization.BaseTransform`."
            self.transformation = transformation
        else:
            self.transformation = visualization.AsinhStretch(
                a=0.01
            ) + visualization.PercentileInterval(99)

        if per_image:
            for i, frame in enumerate(self.images):
                self.images[i] = self.transformation(frame)
        else:
            self.images = self.transformation(self.images)

    def generate_colours(
        self,
        colours: npt.ArrayLike | None = None,
        col_limits: npt.ArrayLike | None = None,
        hsv: bool = False,
    ):
        """
        Generate a colour for each input image.

        Parameters
        ----------
        colours : npt.ArrayLike | None, optional
            _description_, by default None.
        col_limits : [col_min, col_max] | None, optional
            _description_, by default None.
        hsv : bool, optional
            Interpret input colour tuples (a,b,c) as (H,S,V) rather than the
            default (R,G,B). By default False.
        """

        if colours is not None:
            assert (
                len(colours) == self.N
            ), "Number of colours does not match number of input images."
            self.colours = [utils.convert_colour_to_rgba(c, hsv) for c in colours]

        elif col_limits is not None:
            assert (
                len(col_limits) == 2
            ), "`col_limits` must contain exactly two colours."
            self.colours = utils.interpolate_colours(col_limits, self.N, hsv)
        else:
            self.colours = utils.interpolate_colours([(0, 0, 1), (1, 0, 0)], self.N)

    def apply_colours(self):
        """
        _summary_.
        """

        self.rgb_frames = []
        for i, (frame, col) in enumerate(zip(self.images, self.colours)):
            cmap = utils.gen_cmap_from_rgba(col)
            print(col)

            self.rgb_frames.append(cmap(frame))
        self.rgb_frames = np.asarray(self.rgb_frames)

    def combine_to_rgb(self):
        """
        _summary_.
        """
        self.rgb = np.nanmax(self.rgb_frames, axis=0)

    def plot(self):
        """
        _summary_.
        """
        plt.imshow(self.rgb, origin="lower")
        plt.show()
