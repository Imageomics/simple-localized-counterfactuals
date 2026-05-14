import dataclasses
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import trackio
from saev.nn.modeling import MatryoshkaSparseAutoencoder, SparseAutoencoderConfig
from saev.nn.objectives import (
    Matryoshka,
    MatryoshkaObjective,
)
from saev.utils.scheduling import Warmup, WarmupCosine
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from slc.constants import DatasetNames
from slc.datasets import ImageFolderDatasetInterface
from slc.models import ResNet50PytorchRecognitionModel
from slc.utils import get_dataset_root, save_json


class ImageNetFeatureDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.paths = []
        for _, _, files in self.root.walk():
            for f in files:
                self.paths.append(self.root / f)

        # self.paths = self.paths[:1_000]

    def __len__(self):
        # We assume each image had 7 x 7 feature vectors so 49
        return len(self.paths)  # * 7 * 7

    def __getitem__(self, idx):
        path_idx = idx  # // 49
        feature_idx = idx % 49
        random_feat_idx = random.randint(0, 48)

        image_features = np.load(self.paths[path_idx])[random_feat_idx]
        # features = image_features[feature_idx]

        return torch.from_numpy(image_features)


# Training configs
@dataclass
class TrainingConfigs:
    epochs: int = 100
    lr: float = 0.0004
    n_lr_warmup: int = 50
    n_sparsity_warmup: int = 1000
    exp_factor: int = 8
    sparsity_coeff: float = 0.005
    grad_clip: float = 1.0
    n_prefixes: int = 10


def run():
    # Dataset
    root_dset_path = Path(get_dataset_root(DatasetNames.IMAGENET), "images/train")
    print(root_dset_path)
    dset = ImageFolderDatasetInterface(root=root_dset_path)
    train_dset = dset.get_torch_dataset(phase="train")

    # Saving
    save_dir = Path("/local/scratch2/carlyn.1/rcvcf/saes")
    save_dir.mkdir(exist_ok=True, parents=True)

    batch_size = 128
    num_workers = 8

    # Training configs
    cfgs = TrainingConfigs()

    # Intermediate feature save dir
    feature_save_dir = Path("/local/scratch2/carlyn.1/rcvcf/resnet50_features")
    feature_feats_save_dir = Path(
        "/local/scratch2/carlyn.1/rcvcf/resnet50_features/features"
    )
    feature_feats_save_dir.mkdir(exist_ok=True, parents=True)

    classifier = ResNet50PytorchRecognitionModel(
        class_num=1000, use_imagenet_classifier=True
    ).cuda()
    classifier.eval()

    image_feature_dataset = ImageNetFeatureDataset(
        root=Path("/local/scratch2/carlyn.1/rcvcf/resnet50_features/features")
    )

    cfg = SparseAutoencoderConfig(
        d_vit=2048,
        exp_factor=cfgs.exp_factor,
        normalize_w_dec=True,
    )

    sae = MatryoshkaSparseAutoencoder(cfg=cfg).cuda()
    sae.train()

    sae_train_dloader = DataLoader(
        image_feature_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )

    param_groups = [
        {"params": sae.parameters(), "lr": 0.0},
    ]
    optimizer = Adam(param_groups)
    lr_scheduler = WarmupCosine(
        0.0, cfgs.n_lr_warmup, cfgs.lr, len(sae_train_dloader), 0.0
    )
    sparsity_scheduler = Warmup(0.0, cfgs.sparsity_coeff, cfgs.n_sparsity_warmup)

    obj_cfg = Matryoshka(sparsity_coeff=cfgs.sparsity_coeff)
    objective = MatryoshkaObjective(obj_cfg)

    training_tbar = tqdm(
        range(cfgs.epochs * len(sae_train_dloader)),
        desc="Training SAE",
        colour="#883C55",
    )

    trackio.init(
        project="imagenet-sae-training",
        config={
            "epochs": cfgs.epochs * len(sae_train_dloader),
        },
        space_id="Carlyn2015/imagenet-sae-training",
        resume="never",
    )

    epoch_trackers = defaultdict(list)
    tmp_loss_tracker = []
    for epoch in range(cfgs.epochs):
        loss_items = ["loss", "mse", "sparsity", "l0"]
        trackers = defaultdict(list)
        for bi, batch in enumerate(sae_train_dloader):
            batch = batch.reshape(-1, batch.shape[-1])
            sae.normalize_w_dec()

            x_hat, f_x = sae.matryoshka_forward(
                batch.cuda(), n_prefixes=cfgs.n_prefixes
            )
            sae_loss = objective(batch.cuda(), f_x, x_hat)
            sae_loss.loss.backward()
            sae.remove_parallel_grads()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                sae.parameters(), max_norm=cfgs.grad_clip
            )

            optimizer.step()

            optimizer.param_groups[0]["lr"] = lr_scheduler.step()
            objective.sparsity_coeff = sparsity_scheduler.step()

            optimizer.zero_grad()
            for item in loss_items:
                trackers[item].append(getattr(sae_loss, item))

            training_tbar.update(1)
            if bi % 10 == 0:
                training_tbar.set_postfix(
                    {
                        "loss": sae_loss.loss.item(),
                        "l1_loss": sae_loss.l1.item(),
                        "l0_loss": sae_loss.l0.item(),
                        "mse": sae_loss.mse.item(),
                    }
                )

                step = epoch * len(sae_train_dloader)
                step += bi
                trackio.log(
                    {
                        "loss": sae_loss.loss.item(),
                        "l1_loss": sae_loss.l1.item(),
                        "l0_loss": sae_loss.l0.item(),
                        "mse": sae_loss.mse.item(),
                    },
                    step=step,
                )

            if bi % 1000 == 0:
                tmp_loss_tracker.append(sae_loss.loss.item())
                fig, axs = plt.subplots(1, 1, figsize=(16, 9), sharex=True)
                X = np.arange(len(tmp_loss_tracker))
                axs.plot(X, np.array(tmp_loss_tracker))
                fig.savefig(save_dir / "loss-tracking.png")
                plt.close()

        for item in loss_items:
            epoch_trackers[item].append(torch.stack(trackers[item]).mean().item())

        training_tbar.set_postfix({k: v[-1] for k, v in epoch_trackers.items()})

        torch.save(sae.state_dict(), save_dir / "imagenet_sae.pt")

    trackio.finish()
    training_tbar.close()
    torch.save(sae.state_dict(), save_dir / "imagenet_sae.pt")

    X = np.arange(len(epoch_trackers["loss"]))
    fig, axs = plt.subplots(1, 1, figsize=(16, 9), sharex=True)
    for loss_item in ["loss", "mse", "sparsity"]:
        axs.plot(X, np.array(epoch_trackers[loss_item]), label=loss_item)

    l0ax = axs.twinx()
    l0ax.plot(X, np.array(epoch_trackers["l0"]), label="l0", color="green")

    axs.legend()
    l0ax.legend()
    fig.savefig(save_dir / "training-results.png")
    # fig.show()

    save_json(dataclasses.asdict(cfgs), save_dir / "training-configs.json")
    save_json(dataclasses.asdict(cfg), save_dir / "model-configs.json")


if __name__ == "__main__":
    run()
