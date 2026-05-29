import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


# Adapted from: https://github.com/guillaumejs2403/ACE/blob/main/compute_FVA.py
@torch.no_grad()
def compute_FVA(
    oracle_model,
    paired_dataset,  # real and counterfactual pairs
    batch_size=32,
    num_workers=4,
):
    cosine_similarity = torch.nn.CosineSimilarity()

    FVAS = []
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
        cl_feat = oracle_model(cl)
        cf_feat = oracle_model(cf)
        dist = cosine_similarity(cl_feat, cf_feat)
        FVAS.append((dist > 0.5).cpu().numpy())
        dists.append(dist.cpu().numpy())

    return np.concatenate(FVAS), np.concatenate(dists)
