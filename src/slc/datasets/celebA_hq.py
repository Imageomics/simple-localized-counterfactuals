from array import array
from pathlib import Path

import numpy as np
from PIL import Image

from .celebA import CelebADatasetInterface, CelebATorchDataset


class CelebAHQTorchDataset(CelebATorchDataset):
    def __getitem__(self, idx):
        path = self.paths[idx]
        # lbl = self.labels[idx]
        attrs = self.attributes[idx]
        lbl = int(attrs[self.query_label] == 1)

        img_data = np.load(path)[0]
        img_data = np.transpose(img_data, (1, 2, 0))
        img_data = np.clip(img_data, 0, 255)
        image = Image.fromarray(img_data).convert("RGB")

        if self.transforms:
            image = self.transforms(image)

        if self.with_paths:
            return image, lbl, path

        return image, lbl


class CelebAHQDatasetInterface(CelebADatasetInterface):
    def _setup(self) -> None:
        root = Path(self.root)

        self.ids = []
        self.phases = []
        self.image_paths = []
        identities = {}
        phase_map = {0: "train", 1: "val", 2: "test"}
        org_path_to_idx = {}
        with open(root / "image_list.txt", "r") as f:
            for line in f.readlines()[1:]:
                idx, org_idx, org_file_path, _, _ = line.strip().split()
                idx = int(idx)
                org_path_to_idx[org_file_path] = idx

        org_path_to_attributes = {}
        self.attributes = {}
        with open(root / "list_attr_celeba.txt", "r") as f:
            lines = f.readlines()
            self.attribute_names = lines[1].strip().split()
            for line in lines[2:]:
                jpg_image_path, *attrs = line.strip().split()
                attrs = array("i", [int(x) for x in attrs])
                org_path_to_attributes[jpg_image_path] = attrs
                if jpg_image_path not in org_path_to_idx:
                    continue
                id = org_path_to_idx[jpg_image_path]
                self.attributes[id] = attrs

        with open(root / "list_eval_partition.txt", "r") as f:
            for line in f.readlines():
                jpg_image_path, phase = line.strip().split()
                if jpg_image_path not in org_path_to_idx:
                    continue
                id = org_path_to_idx[jpg_image_path]
                self.phases.append(phase_map[int(phase)])
                self.ids.append(id)
                self.image_paths.append(root / f"Img/imgHQ{id:05d}.npy")
                identities[id] = 0  # TODO map this later

        self.class_names = sorted(list(set(identities.values())))

        self.instance_class_names = []
        self.class_idxs = []
        for id in self.ids:
            identity_id = identities[id]
            self.instance_class_names.append(identity_id)
            self.class_idxs.append(identity_id)

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

        return CelebAHQTorchDataset(
            paths=data_dict["paths"],
            labels=data_dict["labels"],
            attributes=data_dict["attributes"],
            transforms=transforms,
            query_label=query_label,
            with_paths=with_paths,
        )
