import numpy as np
from PIL import Image


def create_diff_image(img1, img2, threshold=0.1, make_grey=False):
    diff = np.array(img1).astype(np.float64) - np.array(img2).astype(np.float64)

    if make_grey:
        diff = np.abs(diff).sum(2)
        diff /= diff.max()

        diff[diff < threshold] = 0

        diff = np.stack([diff, diff, diff], axis=2)
        diff_img = Image.fromarray((diff * 255).astype(np.uint8))
    else:
        diff_img_neg = np.zeros_like(np.array(img2)).astype(np.float64)
        diff_img_pos = np.zeros_like(np.array(img2)).astype(np.float64)

        diff_img_neg[diff < 0] = -diff[diff < 0]
        diff_img_pos[diff > 0] = diff[diff > 0]

        diff_img_neg = diff_img_neg.sum(2)
        diff_img_pos = diff_img_pos.sum(2)

        max_val = max(diff_img_neg.max(), diff_img_pos.max())

        diff_img_neg /= max_val
        diff_img_pos /= max_val

        diff_img_neg[diff_img_neg < threshold] = 0
        diff_img_pos[diff_img_pos < threshold] = 0

        diff_img = np.stack(
            [diff_img_pos, np.zeros_like(diff_img_pos), diff_img_neg], axis=2
        )
        diff_img = Image.fromarray((diff_img * 255).astype(np.uint8))
    return diff_img
