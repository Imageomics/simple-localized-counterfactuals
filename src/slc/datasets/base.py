from abc import abstractmethod

from PIL import Image
from torch.utils.data import Dataset


class BasicTorchDataset(Dataset):
    def __init__(self, paths, labels, transforms=None):
        self.paths = paths
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        lbl = self.labels[idx]

        image = Image.open(path).convert("RGB")

        if self.transforms:
            image = self.transforms(image)

        return image, lbl, path


class BasicTorchWithPathsDataset(BasicTorchDataset):
    def __getitem__(self, idx):
        path = self.paths[idx]
        lbl = self.labels[idx]

        image = Image.open(path).convert("RGB")

        if self.transforms:
            image = self.transforms(image)

        return image, lbl, path


class BaseDatasetInterface:
    def __init__(self, root=None):
        self.root = root
        self._setup()

    @abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError(
            f"_setup function is not implemented in class {type(self).__name__}"
        )

    @abstractmethod
    def get_dataset(self, phase="train", class_names=None, class_idxs=None):
        """Should return a dataframe of ids, image paths, class_names, class_idxs

        Args:
            phase (str, optional): _description_. Defaults to "train".
            class_names (_type_, optional): _description_. Defaults to None.
            class_idxs (_type_, optional): _description_. Defaults to None.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError(
            f"get_torch_dataset function is not implemented in class {type(self).__name__}"
        )

    def get_torch_dataset(
        self,
        phase="train",
        class_names=None,
        class_idxs=None,
        transforms=None,
        with_paths=False,
    ):
        data_dict = self.get_dataset(
            phase=phase, class_names=class_names, class_idxs=class_idxs
        )

        if with_paths:
            return BasicTorchWithPathsDataset(
                paths=data_dict["paths"],
                labels=data_dict["labels"],
                transforms=transforms,
            )
        else:
            return BasicTorchDataset(
                paths=data_dict["paths"],
                labels=data_dict["labels"],
                transforms=transforms,
            )

    @abstractmethod
    def get_class_names(self) -> int:
        raise NotImplementedError(
            f"get_class_names function is not implemented in class {type(self).__name__}"
        )

    def get_number_of_classes(self) -> int:
        return len(self.get_class_names())

    def print_classes(self) -> int:
        for i, cn in enumerate(self.get_class_names()):
            print(i, cn)
