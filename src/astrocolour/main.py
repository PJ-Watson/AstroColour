"""Primary module for the AstroColour package."""

from pathlib import Path

import astropy.io.fits as pf
import numpy as np
import numpy.typing as npt


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
    """

    def __init__(
        self,
        images: npt.ArrayLike,
        img_extns: npt.ArrayLike | None = None,
        crop: npt.ArrayLike | None = None,
    ):
        self._validate_images_input(images, img_extns)

    def _validate_images_input(self, images, img_extns):

        if isinstance(images, str | Path):
            self._validate_imgs_path(images, img_extns)
        # elif isinstance
        else:
            self._validate_imgs_array(images)

        pass

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
        assert all(
            [i.shape == self.image_shape for i in self.images]
        ), "The dimensions of the input images do not match."


if __name__ == "__main__":
    with pf.open(
        "/media/sharedData/data/2024_02_14_RPS/cutout_images/A2744_SPIRAL_cutouts_25.0arcsec.fits"
    ) as img_hdul:
        images = []
        UV_img = []
        UV_img.append(img_hdul["F225W_SCI"].data[:500, :])
        UV_img.append(img_hdul["F275W_SCI"].data[:500, :])
        # print (UV_img)
        images.append(UV_img)
        # images.append([img_hdul["F225W_SCI"].data[:500,:]].append(img_hdul["F275W_SCI"].data[:500,:]))
        for l in ["F435W", "F606W", "F814W"]:
            images.append(img_hdul[f"{l}_SCI"].data[:500, :])
        # images = np.asarray(images)
        # print (images)
        ColourImage(images)
