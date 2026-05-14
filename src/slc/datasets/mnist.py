import struct
import logging
from array import array
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from .base import BaseDatasetInterface


class MNISTTorchDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, labels, transforms=None):
        self.imgs = imgs
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image = self.imgs[idx]
        lbl = self.labels[idx]

        image = image.convert("RGB")

        if self.transforms:
            image = self.transforms(image)

        return image, lbl


class MNISTDatasetInterface(BaseDatasetInterface):
    # From: https://www.kaggle.com/code/hojjatk/read-mnist-dataset
    def _read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, "rb") as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(
                    "Magic number mismatch, expected 2049, got {}".format(magic)
                )
            labels = array("B", file.read())

        with open(images_filepath, "rb") as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(
                    "Magic number mismatch, expected 2051, got {}".format(magic)
                )
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols : (i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def _setup(self) -> None:
        root = Path(self.root)

        train_imgs, train_lbls = self._read_images_labels(
            root / "train-images-idx3-ubyte", root / "train-labels-idx1-ubyte"
        )
        train_imgs = np.array(train_imgs)
        train_lbls = np.array(train_lbls)
        test_imgs, test_lbls = self._read_images_labels(
            root / "t10k-images-idx3-ubyte", root / "t10k-labels-idx1-ubyte"
        )
        test_imgs = np.array(test_imgs)
        test_lbls = np.array(test_lbls)

        self.class_names = [str(i) for i in range(10)]  # MNIST has 10 classes (0-9)

        self.ids = []
        self.images = []
        self.instance_class_names = []
        self.class_idxs = []
        self.phases = []

        for id, img, lbl in zip(range(len(train_imgs)), train_imgs, train_lbls):
            self.ids.append(id)
            self.images.append(img)
            self.instance_class_names.append(str(lbl))
            self.class_idxs.append(lbl)
            self.phases.append("train")

        for tid, img, lbl in zip(range(len(test_imgs)), test_imgs, test_lbls):
            self.ids.append(id + 1 + tid)
            self.images.append(img)
            self.instance_class_names.append(str(lbl))
            self.class_idxs.append(lbl)
            self.phases.append("test")

    def get_class_names(self):
        return self.class_names

    def get_dataset(self, phase="train", class_names=None, class_idxs=None):
        logging.warning(
            "For MNIST, directly call get_torch_dataset instead. No transforms passed."
        )
        return self.get_torch_dataset(
            phase=phase, class_names=class_names, class_idxs=class_idxs, transforms=None
        )

    def get_torch_dataset(
        self, phase="train", class_names=None, class_idxs=None, transforms=None
    ):
        if phase not in ["train", "test"]:
            raise ValueError(
                f"{phase} is not a valid phase for the MNIST dataset. Use either 'train', or 'test'."
            )

        if class_idxs is None and class_names is not None:
            class_idxs = []
            for cn in class_names:
                if cn not in self.class_names:
                    raise ValueError(
                        f"{cn} is not a valid class name for the MNIST dataset. Please use from {self.get_class_names()}"
                    )
                class_idxs.append(self.class_names.index(cn))

        if class_idxs is not None:
            min_cls_idx = min(class_idxs)
            max_cls_idx = max(class_idxs)
            if min_cls_idx < 0:
                raise ValueError(f"{min_cls_idx} is not a valid class index")
            if max_cls_idx > max(self.class_idxs):
                raise ValueError(f"{max_cls_idx} is not a valid class index")

        filtered_ids = []
        filtered_images = []
        filtered_class_names = []
        filtered_class_idxs = []
        for id, img, cn, cls_idx, id_phase in zip(
            self.ids,
            self.images,
            self.instance_class_names,
            self.class_idxs,
            self.phases,
        ):
            # Filter by phase
            if phase != id_phase:
                continue

            # Filter by class idx
            if class_idxs is not None and cls_idx not in class_idxs:
                continue

            filtered_ids.append(id)
            filtered_images.append(Image.fromarray(img))
            filtered_class_names.append(cn)
            filtered_class_idxs.append(cls_idx)

        return MNISTTorchDataset(
            imgs=filtered_images,
            labels=filtered_class_idxs,
            transforms=transforms,
        )
