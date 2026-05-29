from argparse import ArgumentParser

import torch
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from torchvision import models

from slc.datasets import Filtered_CF_Paired_Dataset
from slc.evaluation.cout import compute_cout
from slc.models.evaluation import DecisionDensenetModel, DiVEDenseNet121
from slc.models.oracle_celeba_hq_metrics import Normalizer
from slc.utils import load_json

BINARYDATASET = ["celeba", "celeba-hq", "celeba-mv", "bddoia", "bdd100k"]
MULTICLASSDATASETS = ["imagenet"]


def is_binary_dataset(dataset: str) -> bool:
    return dataset in BINARYDATASET


# Adapted from: https://github.com/guillaumejs2403/ACE/blob/main/compute_SimSiamSimilarity.py
def img_transform(img_size):
    def transforms(x):
        x = F.to_tensor(x)
        x = F.resize(
            x, size=[img_size, img_size], interpolation=F.InterpolationMode.BICUBIC
        )
        return x

    return transforms


def run(
    img_label_list: str,
    img_pair_list: str,
    query_label: int = 0,
    target_label: int = 1,
    dataset: str = "imagenet",
    model_weights: str = "",
    num_workers: int = 4,
    batch_size: int = 32,
):
    img_pair_list = load_json(img_pair_list)
    img_label_list = load_json(img_label_list)

    is_binary = is_binary_dataset(dataset)

    if dataset in ["celeba"]:
        img_size = 128
    elif dataset in ["celeba-hq"]:
        img_size = 256
    elif "bdd" in dataset:
        raise NotImplementedError()
    elif dataset in ["imagenet"]:
        img_size = 224
    else:
        raise NotImplementedError()

    paired_dataset = Filtered_CF_Paired_Dataset(  # Filter for originally correct, and query label data
        img_pair_list,
        img_label_list,
        query_label,
        filter_original_correct=True,
        transforms=img_transform(img_size),
        is_binary=is_binary,
    )

    loader = DataLoader(
        paired_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    ql = query_label

    if dataset in ["celeba", "celeba-mv"]:
        classifier = Normalizer(
            DiVEDenseNet121(model_weights, query_label), [0.5] * 3, [0.5] * 3
        ).cuda()

    elif dataset == "celeba-hq":
        assert query_label in [20, 31, 39], (
            "Query label MUST be 20 (Gender), 31 (Smile), or 39 (Gender) for CelebAHQ"
        )
        ql = 0
        if query_label in [31, 39]:
            ql = 1 if query_label == 31 else 2
        classifier = DecisionDensenetModel(3, pretrained=False, query_label=ql)
        classifier.load_state_dict(
            torch.load(model_weights, map_location="cpu")["model_state_dict"]
        )
        classifier = Normalizer(classifier, [0.5] * 3, [0.5] * 3).cuda()

    elif "bdd" in dataset:
        classifier = DecisionDensenetModel(4, pretrained=False, query_label=query_label)
        classifier.load_state_dict(
            torch.load(model_weights, map_location="cpu")["model_state_dict"]
        )
        classifier = Normalizer(classifier, [0.5] * 3, [0.5] * 3).cuda()

    elif "imagenet" in dataset:
        classifier = Normalizer(
            models.resnet50(pretrained=True)
        ).cuda()  # TODO: replace with our interface...
    else:
        assert False, "Need to implement per dataset"
        classifier = Normalizer(models.resnet50(pretrained=True)).cuda()

    classifier.eval()

    results = compute_cout(ql, target_label, classifier, loader, is_binary)
    cout = results[0]

    return cout


if __name__ == "__main__":
    """The dataset needs to filter by correctly classified images only. Additionally, this score is one-way.
    So if we are going from young->old, then that will have a separate score from old->young and have to filtered
    accordingly.
    """
    parser = ArgumentParser()
    parser.add_argument("--img_pair_list", type=str)
    parser.add_argument("--img_label_list", type=str)
    parser.add_argument("--query_label", type=int, default=0)
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="celeba-hq")
    parser.add_argument(
        "--model_weights", type=str, default="weights/checkpoint_0099.pth.tar"
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    cout = run(
        args.img_label_list,
        args.img_pair_list,
        args.query_label,
        args.target_label,
        args.dataset,
        args.model_weights,
        args.num_workers,
        args.batch_size,
    )

    if not is_binary_dataset(args.dataset):
        cout2 = run(
            args.img_label_list,
            args.img_pair_list,
            args.target_label,  # Switch with target
            args.query_label,  # Switch with query
            args.dataset,
            args.model_weights,
            args.num_workers,
            args.batch_size,
        )

        cout = (cout + cout2) / 2.0  # Average the two

    print(f"\nEVAL [COUT: {cout:.4f}]")
