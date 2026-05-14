from argparse import ArgumentParser

import numpy as np
import torchvision.transforms.functional as F

from slc.datasets import CF_Paired_Dataset
from slc.evaluation.mnac import compute_mnac
from slc.models.oracle_celeba_hq_metrics import CelebAHQOracle
from slc.models.oracle_celeba_metrics import CelebAOracle
from slc.utils import load_json


# Adapted from: https://github.com/guillaumejs2403/ACE/blob/main/compute_MNAC.py
def img_transform(x):
    x = F.to_tensor(x)
    x = F.normalize(x, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
    return x


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--img_pair_list", type=str)
    parser.add_argument("--dataset", type=str, choices=["celeba", "celeba-hq"])
    parser.add_argument(
        "--model_weights", type=str, default="weights/checkpoint_0099.pth.tar"
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    img_pair_list = load_json(args.img_pair_list)

    if args.dataset == "celeba":
        oracle = CelebAOracle(weights_path=args.model_weights)
    elif args.dataset == "celeba-hq":
        oracle = CelebAHQOracle(weights_path=args.model_weights)
    else:
        raise NotImplementedError(f"MNAC has not been impleted for {args.dataset}")

    paired_dataset = CF_Paired_Dataset(img_pair_list, transforms=img_transform)

    mnac, _, _ = compute_mnac(
        oracle_model=oracle,
        paired_dataset=paired_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"MNAC score based on {args.img_pair_list}:")
    print(f"Mean: {np.mean(mnac)}")
    print(f"STD: {np.std(mnac)}")
