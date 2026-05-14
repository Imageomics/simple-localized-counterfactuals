# Consider using https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.ImageNet.html
import logging
from pathlib import Path

from PIL import Image
from torchvision.datasets import ImageNet

from slc.utils import load_json

from .base import BaseDatasetInterface


class ImageNetDatasetInterface(BaseDatasetInterface):
    def _setup(self) -> None:
        self.class_data = load_json(Path(self.root, "imagenet_class_index.json"))
        self.class_names = [self.class_data[str(i)][1] for i in range(1000)]

    def get_class_names(self):
        return self.class_names

    def get_dataset(self, phase="train", class_names=None, class_idxs=None):
        logging.warning(
            "Imagenet dataset doesn't filter by class_name or class_idx at the moment"
        )
        pass

    def get_torch_dataset(
        self, phase="train", class_names=None, class_idxs=None, transforms=None
    ):
        if phase not in ["train", "val"]:
            raise ValueError(
                f"{phase} is not a valid phase for the ImageNet dataset. Use either 'train' or 'val'."
            )
        return ImageNet(
            self.root,
            split=phase,
            transform=transforms,
            loader=lambda x: Image.open(x).convert("RGB"),
        )
