# Note: this is with LPIPS
# It is used as the average LPIPS distance between the same counterfactual instances across separate independent runs.
# It is trying to measure, Do your counterfactuals vary in diversity between runs??
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


# Adapted from: https://github.com/guillaumejs2403/ACE/blob/main/compute_LPIPS.py
@torch.inference_mode()
def compute_lpips_diversity(
    LPIPS,
    paired_dataset,  # counterfactuals of the same instance across different experiments
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

    dists = []

    for cfs in tqdm(loader):
        dist = []

        for i in range(len(cfs)):
            cf1 = cfs[i].to(dtype=torch.float).cuda()

            for j in range(i + 1, len(cfs)):
                cf2 = cfs[j].to(dtype=torch.float).cuda()
                dist.append(
                    LPIPS.forward(cf1, cf2, normalize=False)
                )  # data is already in the [-1,1] range

        dists.append(sum(dist) / len(dist))

    return torch.cat(dists).cpu().detach().numpy()
