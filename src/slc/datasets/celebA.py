from array import array
from pathlib import Path

import numpy as np
from PIL import Image

from .base import BaseDatasetInterface, BasicTorchDataset


class CelebATorchDataset(BasicTorchDataset):
    def __init__(
        self, paths, labels, attributes, query_label, transforms=None, with_paths=False
    ):
        self.paths = paths
        self.labels = labels
        self.transforms = transforms
        self.attributes = attributes
        self.query_label = query_label
        self.with_paths = with_paths

    def __getitem__(self, idx):
        path = self.paths[idx]
        # lbl = self.labels[idx]
        attrs = self.attributes[idx]
        lbl = int(attrs[self.query_label] == 1)

        image = Image.open(path).convert("RGB")

        if self.transforms:
            image = self.transforms(image)

        if self.with_paths:
            return image, lbl, path

        return image, lbl


class CelebADatasetInterface(BaseDatasetInterface):
    def _setup(self) -> None:
        root = Path(self.root)

        self.ids = []
        self.phases = []
        self.image_paths = []
        phase_map = {0: "train", 1: "val", 2: "test"}
        with open(root / "Eval/list_eval_partition.txt", "r") as f:
            for line in f.readlines():
                jpg_image_path, phase = line.strip().split()
                id, _ = jpg_image_path.split(".")
                self.phases.append(phase_map[int(phase)])
                id = int(id)
                self.ids.append(int(id))
                self.image_paths.append(root / f"Img/{id:06d}.png")

        identities = {}
        with open(root / "Anno/identity_CelebA.txt", "r") as f:
            for line in f.readlines():
                jpg_image_path, identity_id = line.strip().split()
                id, _ = jpg_image_path.split(".")
                id = int(id)
                identities[int(id)] = int(identity_id)

        self.class_names = sorted(list(set(identities.values())))

        self.attributes = {}
        with open(root / "Anno/list_attr_celeba.txt", "r") as f:
            lines = f.readlines()
            self.attribute_names = lines[1].strip().split()
            for line in lines[2:]:
                jpg_image_path, *attrs = line.strip().split()
                id, _ = jpg_image_path.split(".")
                id = int(id)
                self.attributes[int(id)] = array("i", [int(x) for x in attrs])

        self.instance_class_names = []
        self.class_idxs = []
        for id in self.ids:
            identity_id = identities[id]
            self.instance_class_names.append(identity_id)
            self.class_idxs.append(identity_id)

    def get_class_names(self):
        return self.class_names

    def get_attribute_names(self):
        return self.attribute_names

    def get_dataset(self, phase="train", class_names=None, class_idxs=None):
        if phase not in ["train", "val", "test"]:
            raise ValueError(
                f"{phase} is not a valid phase for the Oxford Flowers dataset. Use either 'train', 'val' or 'test'."
            )

        if class_idxs is None and class_names is not None:
            class_idxs = []
            for cn in class_names:
                if cn not in self.class_names:
                    raise ValueError(
                        f"{cn} is not a valid class name for the Oxford Flowers dataset. Please use from {self.get_class_names()}"
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
        filtered_paths = []
        filtered_class_names = []
        filtered_class_idxs = []
        filtered_attributes = []
        for id, path, cn, cls_idx, id_phase in zip(
            self.ids,
            self.image_paths,
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
            filtered_paths.append(path)
            filtered_class_names.append(cn)
            filtered_class_idxs.append(cls_idx)
            filtered_attributes.append(np.array(self.attributes[id]))

        return {
            "ids": filtered_ids,
            "paths": filtered_paths,
            "class_names": filtered_class_names,
            "labels": filtered_class_idxs,
            "attributes": filtered_attributes,
            "attribute_names": self.attribute_names,
        }

    def get_torch_dataset(
        self,
        phase="train",
        class_names=None,
        class_idxs=None,
        query_label=None,
        transforms=None,
        with_paths=False,
    ):
        assert query_label is not None, (
            "Please provide a query label for the CelebA dataset"
        )

        data_dict = self.get_dataset(
            phase=phase, class_names=class_names, class_idxs=class_idxs
        )

        return CelebATorchDataset(
            paths=data_dict["paths"],
            labels=data_dict["labels"],
            attributes=data_dict["attributes"],
            transforms=transforms,
            query_label=query_label,
            with_paths=with_paths,
        )
