from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F

from slc.datasets import CF_Paired_Dataset
from slc.evaluation.sim_siam_similarity import compute_sim_siam_similarity
from slc.utils import load_json


# Adapted from: https://github.com/guillaumejs2403/ACE/blob/main/eval_utils/simsiam.py
class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """

    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        self.criterion = nn.CosineSimilarity(dim=1)

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),  # first layer
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),  # second layer
            self.encoder.fc,
            nn.BatchNorm1d(dim, affine=False),
        )  # output layer
        self.encoder.fc[
            6
        ].bias.requires_grad = False  # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),  # hidden layer
            nn.Linear(pred_dim, dim),
        )  # output layer

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.encoder(x1)  # NxC
        z2 = self.encoder(x2)  # NxC

        # p1 = self.predictor(z1) # NxC
        # p2 = self.predictor(z2) # NxC

        # dist = (self.criterion(p1, z2) + self.criterion(p2, z1)) * 0.5
        dist = self.criterion(z1, z2)

        return dist


def get_simsiam_dist(weights_path):
    import torchvision.models as models

    model = SimSiam(models.resnet50, dim=2048, pred_dim=512)
    state_dict = torch.load(weights_path, map_location="cpu")["state_dict"]
    model.load_state_dict({k[7:]: v for k, v in state_dict.items()})
    return model


# Adapted from: https://github.com/guillaumejs2403/ACE/blob/main/compute_SimSiamSimilarity.py
def img_transform(x):
    x = np.array(x, dtype=np.uint8)
    x = x.astype(np.float32) / 255
    x = torch.from_numpy(x).float()
    x = x.permute((2, 0, 1))  # C x H x W
    x = F.normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=True)
    return x


def run(
    img_pair_list: str,
    model_weights: str = "",
    num_workers: int = 4,
    batch_size: int = 32,
):
    img_pair_list = load_json(img_pair_list)

    oracle_model = get_simsiam_dist(model_weights).cuda()
    oracle_model.eval()

    paired_dataset = CF_Paired_Dataset(img_pair_list, transforms=img_transform)

    dists = compute_sim_siam_similarity(
        oracle_model,
        paired_dataset,  # real and counterfactual pairs
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return np.mean(dists).item()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--img_pair_list", type=str)
    parser.add_argument(
        "--model_weights", type=str, default="weights/checkpoint_0099.pth.tar"
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    sss = run(
        args.img_pair_list,
        args.model_weights,
        args.num_workers,
        args.batch_size,
    )

    print(f"SSS score based on {args.img_pair_list}: {sss:>4f}")
