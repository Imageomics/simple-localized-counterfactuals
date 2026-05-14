from pathlib import Path

import scipy.io as sio

from .base import BaseDatasetInterface


class FlowersDatasetInterface(BaseDatasetInterface):
    def _setup(self) -> None:
        root = Path(self.root)
        image_labels = sio.loadmat(root / "imagelabels.mat")["labels"][0]
        image_labels = [int(x) - 1 for x in image_labels]
        setid = sio.loadmat(root / "setid.mat")

        with open(root / "label_names.txt", "r") as f:
            cls_idx_to_name_map = {
                i: line.strip()[1:-1] for i, line in enumerate(f.readlines())
            }

        self.class_names = [
            cls_idx_to_name_map[i] for i in range(len(cls_idx_to_name_map))
        ]

        train_ids = [int(x) for x in setid["trnid"][0]]
        val_ids = [int(x) for x in setid["valid"][0]]
        test_ids = [int(x) for x in setid["tstid"][0]]

        self.ids = []
        self.image_paths = []
        self.instance_class_names = []
        self.class_idxs = []
        self.phases = []

        for id_set, phase in zip(
            [train_ids, val_ids, test_ids],
            ["train", "val", "test"],
        ):
            for id in id_set:
                self.ids.append(id)
                self.image_paths.append(root / "jpg" / f"image_{id:05d}.jpg")
                self.instance_class_names.append(
                    cls_idx_to_name_map[image_labels[id - 1]]
                )  # 1 -indexed
                self.class_idxs.append(image_labels[id - 1])
                self.phases.append(phase)

    def get_class_names(self):
        return self.class_names

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

        return {
            "ids": filtered_ids,
            "paths": filtered_paths,
            "class_names": filtered_class_names,
            "labels": filtered_class_idxs,
        }
