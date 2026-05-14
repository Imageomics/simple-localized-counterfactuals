import pickle
from argparse import ArgumentParser

import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.models import resnet50

from slc.datasets import CF_Paired_Dataset
from slc.evaluation.fva import compute_FVA
from slc.utils import load_json


# Adapted from: https://github.com/guillaumejs2403/ACE/blob/main/eval_utils/resnet50_facevgg2_FVA.py
def load_state_dict(model, fname):
    """
    Set parameters converted from Caffe models authors of VGGFace2 provide.
    See https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/.
    Arguments:
        model: model
        fname: file name of parameters converted from a Caffe model, assuming the file format is Pickle.
    """
    with open(fname, "rb") as f:
        weights = pickle.load(f, encoding="latin1")

    own_state = model.state_dict()
    for name, param in weights.items():
        if name in own_state:
            try:
                own_state[name].copy_(torch.from_numpy(param))
            except Exception:
                raise RuntimeError(
                    "While copying the parameter named {}, whose dimensions in the model are {} and whose "
                    "dimensions in the checkpoint are {}.".format(
                        name, own_state[name].size(), param.size()
                    )
                )
        else:
            print(name, "not in the state dict")

    model.load_state_dict(own_state)


# Adapted from: https://github.com/guillaumejs2403/ACE/blob/main/compute_FVA.py
def img_transform(x):
    x = F.resize(x, size=[224, 224])
    x = np.array(x)
    x = x[:, :, ::-1]  # RGB -> BGR
    x = x.astype(np.float32)
    mean_bgr = np.array([91.4953, 103.8827, 131.0912])
    x -= mean_bgr
    x = x.transpose(2, 0, 1)  # C x H x W
    x = torch.from_numpy(x).float()
    return x


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--img_pair_list", type=str)
    parser.add_argument(
        "--model_weights", type=str, default="weights/resnet50_ft_weight.pkl"
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    img_pair_list = load_json(args.img_pair_list)

    oracle_model = resnet50(num_classes=8631).cuda()
    load_state_dict(oracle_model, args.model_weights)
    oracle_model.eval()

    def new_forward(x):
        x = oracle_model.conv1(x)
        x = oracle_model.bn1(x)
        x = oracle_model.relu(x)
        x = oracle_model.maxpool(x)

        x = oracle_model.layer1(x)
        x = oracle_model.layer2(x)
        x = oracle_model.layer3(x)
        x = oracle_model.layer4(x)

        x = oracle_model.avgpool(x)

        return x

    oracle_model.forward = new_forward

    paired_dataset = CF_Paired_Dataset(img_pair_list, transforms=img_transform)

    fva, dists = compute_FVA(
        oracle_model,
        paired_dataset,  # real and counterfactual pairs
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"FVA score based on {args.img_pair_list}:")
    print("FVA", np.mean(fva))
    print("FVA (STD)", np.std(fva))
    print("mean dist", np.mean(dists))
    print("std dist", np.std(dists))
