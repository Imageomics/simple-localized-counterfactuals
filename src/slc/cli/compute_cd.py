from argparse import ArgumentParser

import numpy as np
import torchvision.transforms.functional as F

from slc.datasets import CF_Paired_Dataset
from slc.evaluation.cd import compute_CorrMetric, get_correlations
from slc.models.oracle_celeba_hq_metrics import CelebAHQOracle
from slc.models.oracle_celeba_metrics import CelebAOracle
from slc.utils import load_json


def img_transform(x):
    x = F.to_tensor(x)
    x = F.normalize(x, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
    return x


# Adapted from: https://github.com/guillaumejs2403/ACE/blob/6684b93ef2d2d6a594fdbac87db25e3f640340e7/compute_CD.py
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--img_pair_list", type=str)
    parser.add_argument("--celeba_path", type=str)
    parser.add_argument("--query_label", type=int)
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

    corrs, labels = get_correlations(
        args.celeba_path, args.query_label, args.dataset == "celeba-hq"
    )

    sorted = np.argsort(np.abs(corrs))[::-1]

    results = compute_CorrMetric(
        oracle,
        paired_dataset,
        args.query_label,
        diff=True,
        remove_unchanged_oracle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print("CD Result:", np.sum(np.abs(results[sorted] - corrs[sorted])))
