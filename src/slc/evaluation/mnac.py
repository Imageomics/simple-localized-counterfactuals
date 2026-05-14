# This is only computed for the CelebA and CelebA-HQ datasets
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


# Adapted from: https://github.com/guillaumejs2403/ACE/blob/main/compute_MNAC.py
@torch.inference_mode()
def compute_mnac(
    oracle_model,
    paired_dataset,  # real and counterfactual pairs
    batch_size=32,
    num_workers=4,
):
    MNACS = []
    dists = []
    loader = DataLoader(
        paired_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    for cl, cf in tqdm(loader):
        d_cl = oracle_model(cl.to(dtype=torch.float).cuda())
        d_cf = oracle_model(cf.to(dtype=torch.float).cuda())
        MNACS.append(((d_cl > 0.5) != (d_cf > 0.5)).sum(dim=1).cpu().numpy())
        dists.append([d_cl.cpu().numpy(), d_cf.cpu().numpy()])

    return (
        np.concatenate(MNACS),
        np.concatenate([d[0] for d in dists]),
        np.concatenate([d[1] for d in dists]),
    )
