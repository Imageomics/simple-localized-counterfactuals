from pathlib import Path

from slc.utils import read_deliminted_data

from .base import BaseDatasetInterface


class CUBDatasetInterface(BaseDatasetInterface):
    def _setup(self) -> None:
        self.ids = []
        self.image_paths = []
        self.instance_class_names = []
        self.class_idxs = []
        self.is_training = []

        root_dir = Path(self.root)

        def create_two_column_map(fp):
            return {key: value for key, value in read_deliminted_data(fp, delim=" ")}

        id_local_path_map = create_two_column_map(root_dir / "images.txt")

        id_training_map = create_two_column_map(root_dir / "train_test_split.txt")

        # NOTE class id is 1-200 (need to subtract one)
        id_class_map = create_two_column_map(root_dir / "image_class_labels.txt")
        cls_id_cls_name_map = {}
        with open(root_dir / "classes.txt", "r") as f:
            for line in f:
                class_idx, *class_name = line.strip().split()
                class_name = " ".join(class_name)
                cls_id_cls_name_map[class_idx] = class_name

        self.class_names = [x.split(".")[1] for x in cls_id_cls_name_map.values()]

        for id, local_path in id_local_path_map.items():
            self.ids.append(id)
            self.image_paths.append(Path(root_dir, "images", local_path))
            cls_idx = id_class_map[id]
            cls_name = cls_id_cls_name_map[cls_idx]

            self.class_idxs.append(int(cls_idx) - 1)
            self.instance_class_names.append(cls_name)

            self.is_training.append(id_training_map[id])

    def get_class_names(self):
        return self.class_names

    def get_dataset(self, phase="train", class_names=None, class_idxs=None):
        if phase not in ["train", "test"]:
            raise ValueError(
                f"{phase} is not a valid phase for the CUB dataset. Use either 'train' or 'test'."
            )

        if class_idxs is None and class_names is not None:
            class_idxs = []
            for cn in class_names:
                if cn not in self.class_names:
                    raise ValueError(
                        f"{cn} is not a valid class name for the CUB dataset. Please use from {self.get_class_names()}"
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
        for id, path, cn, cls_idx, is_train in zip(
            self.ids,
            self.image_paths,
            self.instance_class_names,
            self.class_idxs,
            self.is_training,
        ):
            # Filter by phase
            if phase == "train" and int(is_train) != 1:
                continue
            if phase == "test" and int(is_train) != 0:
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
