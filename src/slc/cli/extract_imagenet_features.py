from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from slc.constants import DatasetNames
from slc.datasets import ImageFolderDatasetInterface
from slc.models import ResNet50PytorchRecognitionModel
from slc.utils import get_dataset_root


# Training configs
@dataclass
class TrainingConfigs:
    epochs: int = 300
    lr: float = 0.001
    n_lr_warmup: int = 50
    n_sparsity_warmup: int = 0
    exp_factor: int = 8
    sparsity_coeff: float = 0.0004
    grad_clip: float = 1.0


def run():
    # Dataset
    root_dset_path = Path(get_dataset_root(DatasetNames.IMAGENET), "images/train")
    print(root_dset_path)
    dset = ImageFolderDatasetInterface(root=root_dset_path)
    train_dset = dset.get_torch_dataset(phase="train")

    # Saving
    save_dir = Path("/local/scratch2/carlyn.1/rcvcf/saes")
    save_dir.mkdir(exist_ok=True, parents=True)

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

    def collate_fn(batch):
        features = []
        labels = []
        for img, label, path in batch:
            feats = classifier.preprocess_image(img)
            features.append(feats)
            labels.append(torch.tensor(label))
        features = torch.stack(features)
        labels = torch.stack(labels)

        return features, labels

    train_dloader = DataLoader(
        train_dset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    image_features = []
    image_labels = []
    idx = 0
    for images, labels in tqdm(
        train_dloader,
        desc="Extracting and saving features for SAE training",
        colour="#5AF028",
    ):
        with torch.no_grad():
            feats = classifier.forward_features_spatial(images.cuda()).detach().cpu()
        for ft, lbl in zip(feats, labels):
            ft = torch.permute(ft, (1, 2, 0))
            ft = ft.reshape(-1, ft.shape[2]).numpy()
            np.save(feature_feats_save_dir / f"{idx}_features.npy", ft)
            image_labels.append(lbl)
            idx += 1

    np.save(feature_save_dir / "labels.npy", np.array(image_labels))


if __name__ == "__main__":
    run()
