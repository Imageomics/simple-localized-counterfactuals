from argparse import ArgumentParser

import lpips
import numpy as np
import torchvision.transforms.functional as F

from slc.datasets import CF_Paired_Dataset
from slc.evaluation.diversity import compute_lpips_diversity
from slc.utils import load_json


# Adapted from: https://github.com/guillaumejs2403/ACE/blob/main/compute_LPIPS.py
def img_transform(x):
    x = F.to_tensor(x)
    x = F.normalize(x, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
    return x


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--img_pair_list", type=str)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    img_pair_list = load_json(args.img_pair_list)

    LPIPS = lpips.LPIPS(net="vgg", spatial=False).cuda()
    LPIPS.eval()

    paired_dataset = CF_Paired_Dataset(img_pair_list, transforms=img_transform)

    res = compute_lpips_diversity(
        LPIPS,
        paired_dataset,  # real and counterfactual pairs
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"LPIPS Diversity score based on {args.img_pair_list}: {np.mean(res).item()}")
