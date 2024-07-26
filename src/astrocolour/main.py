"""Primary module for the AstroColour package."""

from pathlib import Path

import astropy.io.fits as pf
import astropy.visualization as visualization
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from astropy.stats import sigma_clipped_stats
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv

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
        # transformation=None,
        transformation_kwargs=None,
        colours=None,
        colours_kwargs=None,
    ):
        self._validate_images_input(images, img_extns)

        if transformation_kwargs is None:
            transformation_kwargs = {}
        self.apply_transformation(**transformation_kwargs)

        if colours_kwargs is None:
            colours_kwargs = {}
        self.generate_colours(colours, **colours_kwargs)

        self.apply_colours()
        self.combine_to_rgb()
        # self.plot()

    def __array__(self):
        return self.rgb

    def __getitem__(self, key):
        """
        Slice the RGB image.
        """
        if (
            isinstance(key, tuple)
            and len(key) == 2
            and all(
                isinstance(key[i], slice) and (key[i].start != key[i].stop)
                for i in (0, 1)
            )
        ):
            return self.rgb[key]
        else:
            raise TypeError(f"{key!r} is not a valid 2D slice object")

    def _validate_images_input(self, images, img_extns):

        if isinstance(images, str | Path):
            self._validate_imgs_path(images, img_extns)
        else:
            self._validate_imgs_array(images)

    def _validate_imgs_array(self, images):
        """
        Validates the image input, if given as an array-like.

        Checks that the N input images all have the same shape (i x j), and creates
        an N x i x j array to store them.
        """

        self.images = []
        for img in images:
            img_view = np.atleast_2d(img)
            if img_view.ndim == 2:
                self.images.append(
                    np.asarray(img).reshape(img_view.shape[0], img_view.shape[1])
                )
            if img_view.ndim > 2:
                self.images.append(
                    sigma_clipped_stats(img_view, axis=0)[1].reshape(
                        img_view.shape[1], img_view.shape[2]
                    )
                )

        self.image_shape = self.images[0].shape
        self.N = len(self.images)

        assert all(
            [i.shape == self.image_shape for i in self.images]
        ), "The dimensions of the input images do not match."

        self.images = np.asarray(self.images)

    def apply_transformation(
        self,
        stretch: visualization.BaseStretch | None = visualization.AsinhStretch(a=1e-2),
        interval: visualization.BaseInterval | None = visualization.PercentileInterval(
            99.9
        ),
        transformation: visualization.BaseTransform | None = None,
        per_image: bool = False,
    ):
        """
        Apply a transformation (interval and stretch) to the images.

        Parameters
        ----------

        stretch : visualization.BaseStretch | None, optional
            The stretch to be applied to the image. This defaults to an Asinh stretch, with
            the logarithmic transition occurring at 1% of the normalised maximum intensity.
        interval : visualization.BaseInterval | None, optional
            The interval within which the image will be normalised. By default, this is
            99% of the full intensity range.
        transformation : visualization.BaseTransform | None, optional
            Any transformation to be applied to the image. This overrides all other
            options (`stretch` and `interval`) if supplied, by default None.
        per_image : bool, optional
            Apply the transformation to each input image individually. This may result
            in inconsistent scaling if using percentile intervals, and is False by default.
        """

        if transformation is not None:
            assert isinstance(
                transformation, visualization.BaseTransform
            ), "`transformation` must be an instance of `astropy.visualization.BaseTransform`."
            self.transformation = transformation
        elif stretch is not None and interval is not None:
            self.transformation = stretch + interval
        elif stretch is not None and interval is None:
            self.transformation = stretch
        elif stretch is None and interval is not None:
            self.transformation = interval
        else:
            # Should this be a warning instead?
            raise ValueError("No valid transformation supplied.")

        if per_image:
            for i, frame in enumerate(self.images):
                self.images[i] = self.transformation(frame, clip=False)
        else:
            self.images = self.transformation(self.images, clip=False)

    def generate_colours(
        self,
        colours: npt.ArrayLike | None = None,
        col_limits: npt.ArrayLike | None = None,
        hsv: bool = False,
        **kwargs,
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
        **kwargs : dict, optional
            Other arguments to pass to `utils.interpolate_colours`.
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
            self.colours = utils.interpolate_colours(col_limits, self.N, hsv, **kwargs)
        else:
            self.colours = utils.interpolate_colours(
                [(0, 0, 1), (1, 0, 0)], self.N, **kwargs
            )

    def apply_colours(self):
        """
        _summary_.
        """

        self.rgb_frames = []
        for i, (frame, col) in enumerate(zip(self.images, self.colours)):
            cmap = utils.gen_cmap_from_rgba(col)
            # print(col)

            self.rgb_frames.append(cmap(frame))
        self.rgb_frames = np.asarray(self.rgb_frames)
        self.rgb_frames_scaling = np.array([1.0, 1.0, 1.0])

    def combine_to_rgb(self, method="sum"):
        """
        _summary_.

        Parameters
        ----------
        method : str, optional
            _description_, by default "sum".
        """
        match method:
            case "max":
                self.rgb = np.nanmax(self.rgb_frames, axis=0)
            case "sum":
                self.rgb = (
                    np.nansum(self.rgb_frames, axis=0)
                    / np.nansum(self.colours, axis=0)[np.newaxis, np.newaxis, :]
                )

        self.rgb_scaling = np.array([1.0, 1.0, 1.0])

    def plot(self, ax=None, individual=False):
        """
        _summary_.

        Parameters
        ----------
        ax : _type_, optional
            _description_, by default None.
        individual : bool, optional
            _description_, by default False.

        Returns
        -------
        fig, ax
            _description_.
        """
        if individual:
            columns = np.ceil(np.sqrt(self.N)).astype(int)
            rows = np.ceil(self.N / columns).astype(int)

            fig, axs_2d = plt.subplots(rows, columns, sharex=True, sharey=True)
            axs = axs_2d.flatten()

            for i, f in enumerate(self.rgb_frames):
                axs[i].imshow(f, origin="lower")
            return fig, axs_2d
        else:
            if ax is not None:
                fig = ax.get_figure()
            else:
                fig, ax = plt.subplots()
            ax.imshow(self.rgb, origin="lower")
            return fig, ax
        # plt.show()
        # return

    def set_white_balance(self, centre, radius, method="sigma_clip"):
        """
        _summary_.

        Parameters
        ----------
        centre : _type_
            _description_.
        radius : _type_
            _description_.
        method : str, optional
            _description_, by default "sigma_clip".
        """

        rr, cc = utils.circle_coords(centre, radius, self.image_shape)
        # print(self.rgb.shape)
        # print(self.rgb[rr, cc, :3].shape)
        match method:
            case "median":
                avg_fn = np.nanmedian
            case "mean":
                avg_fn = np.nanmean
            case "sigma_clip":
                avg_fn = sigma_clipped_stats

        self.rgb[:, :, :3] /= self.rgb_scaling

        rgb_current = avg_fn(self.rgb[rr, cc, :3], axis=0)
        if method == "sigma_clip":
            rgb_current = rgb_current[0]
        # rgb_current = sigma_clip(self.rgb[rr,cc,:3],axis=0)
        # print (rgb_current)
        rgb_current[rgb_current == 0] = np.nanmedian(rgb_current[rgb_current != 0])
        # print (rgb_current)
        # print(np.nanmean(rgb_current) / rgb_current)
        # print (np.nanmin(self.rgb[rr,cc,:3]))
        self.rgb_scaling = np.nanmean(rgb_current) / rgb_current
        self.rgb[:, :, :3] *= self.rgb_scaling
        # for i, scale in enumerate(np.nanmean(rgb_current) / rgb_current):
        #     # for i, scale in enumerate(rgb_current/np.nanmean(rgb_current)):
        #     self.rgb[:, :, i] *= scale

    def adjust_contrast(self, cutoff=0.5, gain=10, inv=False):
        """
        _summary_.

        Parameters
        ----------
        cutoff : float, optional
            _description_, by default 0.5.
        gain : int, optional
            _description_, by default 10.
        inv : bool, optional
            _description_, by default False.
        """

        # #skimage.exposure.adjust_sigmoid
        # if cutoff is None:
        #     cutoff = np.nanmedian(self.rgb[:,:,:-1])

        # scale = float(dtype_limits(image, True)[1] - dtype_limits(image, True)[0])
        scale = 1.0

        if inv:
            self.rgb[:, :, :-1] = (
                1 - 1 / (1 + np.exp(gain * (cutoff - self.rgb[:, :, :-1] / scale)))
            ) * scale

        else:
            self.rgb[:, :, :-1] = (
                1 / (1 + np.exp(gain * (cutoff - self.rgb[:, :, :-1] / scale)))
            ) * scale

    def adjust_saturation(self, factor=1.25):
        """
        _summary_.

        Parameters
        ----------
        factor : float, optional
            _description_, by default 1.25.
        """

        hsv_copy = rgb_to_hsv(self.rgb[:, :, :-1])
        hsv_copy[:, :, 1] *= factor
        self.rgb[:, :, :-1] = hsv_to_rgb(hsv_copy)

        # # #skimage.exposure.adjust_sigmoid
        # # if cutoff is None:
        # #     cutoff = np.nanmedian(self.rgb[:,:,:-1])

        # # scale = float(dtype_limits(image, True)[1] - dtype_limits(image, True)[0])
        # scale = 1.

        # if inv:
        #     self.rgb[:,:,:-1] = (1 - 1 / (1 + np.exp(gain * (cutoff - self.rgb[:,:,:-1] / scale)))) * scale

        # else:
        #     self.rgb[:,:,:-1] = (1 / (1 + np.exp(gain * (cutoff - self.rgb[:,:,:-1] / scale)))) * scale
