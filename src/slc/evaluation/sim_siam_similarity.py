import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


# Adapted from: https://github.com/guillaumejs2403/ACE/blob/main/compute_SimSiamSimilarity.py
@torch.inference_mode()
def compute_sim_siam_similarity(
    oracle_model,
    paired_dataset,  # real and counterfactual pairs
    batch_size=32,
    num_workers=4,
):
    dists = []
    loader = DataLoader(
        paired_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    for cl, cf in tqdm(loader):
        cl = cl.to(dtype=torch.float).cuda()
        cf = cf.to(dtype=torch.float).cuda()
        dists.append(oracle_model(cl, cf).cpu().numpy())

    return np.concatenate(dists)
