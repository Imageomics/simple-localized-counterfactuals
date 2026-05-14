from typing import Optional

import numpy as np
from PIL import Image


def make_image_grid(
    imgs, nrows: int = 1, ncols: Optional[int] = None, img_size: int = 128
) -> Image.Image:
    if ncols is None:
        ncols = len(imgs) // nrows + int(len(imgs) % nrows > 0)

    grid_image = np.zeros((nrows * img_size, ncols * img_size, 3), dtype=np.uint8)
    for i, img in enumerate(imgs):
        row = i // ncols
        col = i % ncols
        img = img.resize((img_size, img_size), resample=Image.Resampling.BICUBIC)
        grid_image[
            row * img_size : (row + 1) * img_size,
            col * img_size : (col + 1) * img_size,
        ] = np.array(img)
    return grid_image
