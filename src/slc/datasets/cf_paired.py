from PIL import Image
from torch.utils.data import Dataset


class CF_Paired_Dataset(Dataset):
    def __init__(self, img_pair_list, transforms=None):
        self.img_pair_list = img_pair_list
        self.transforms = transforms

    def __len__(self):
        return len(self.img_pair_list)

    def __getitem__(self, idx):
        imgs = []
        for path in self.img_pair_list[idx]:
            x = Image.open(path).convert("RGB")
            if self.transforms:
                x = self.transforms(x)

            imgs.append(x)
        return imgs


class Filtered_CF_Paired_Dataset(Dataset):
    def __init__(
        self,
        img_pair_list,
        img_label_list,
        query_label,
        filter_original_correct=False,
        transforms=None,
        is_binary=False,
    ):
        self.img_pair_list = img_pair_list
        self.img_label_list = img_label_list
        new_img_pair_list = []
        new_img_label_list = []
        idx_to_pair = {
            int(y.split("/")[-1].split(".")[0]): [x, y] for x, y in self.img_pair_list
        }
        idx_set = set()
        for i, label_info in enumerate(self.img_label_list):
            if filter_original_correct and int(label_info["org_pred"]) != int(
                label_info["src"]
            ):
                continue

            if not is_binary and label_info["src"] != query_label:
                continue

            if is_binary:
                # This is needed since we do multi processing to speed up generation, but duplicates can occur and the order
                # may be off. We started doing this when evaluating binary datasets, so that's why we only check here.
                idx = int(label_info["idx"])
                if idx in idx_set:
                    continue
                idx_set.add(idx)

                if idx not in idx_to_pair:
                    print(f"Missing data at index: {idx}")
                    continue
                new_img_pair_list.append(idx_to_pair[idx])
            else:
                new_img_pair_list.append(self.img_pair_list[i])
            new_img_label_list.append(self.img_label_list[i])

        # print(idx_to_pair.keys())
        self.img_pair_list = new_img_pair_list
        self.img_label_list = new_img_label_list

        self.query_label = query_label
        self.transforms = transforms

    def __len__(self):
        return len(self.img_pair_list)

    def __getitem__(self, idx):
        imgs = []
        for path in self.img_pair_list[idx]:
            x = Image.open(path).convert("RGB")
            if self.transforms:
                x = self.transforms(x)
            imgs.append(x)
        return imgs
