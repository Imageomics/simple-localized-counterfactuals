import os
import random
import shutil
from pathlib import Path

import torch
from pytorch_fid.fid_score import calculate_fid_given_paths
from tqdm import tqdm


def compute_fid(real_dir, gen_dir, num_workers=4, batch_size=32, dims=2048):
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

    fid_value = calculate_fid_given_paths(
        [real_dir, gen_dir], batch_size, device, dims, num_workers
    )

    return fid_value


def compute_sfid(
    img_pair_list,  # list of image path pairs were first item is associated with real_dir and second item with gen_dir
    tmp_dir,
    iterations=10,
    seed=2025,
    num_workers=4,
    batch_size=32,
    dims=2048,
):
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

    tmp_dir = Path(tmp_dir)

    random.seed(seed)
    fids = []
    for _ in tqdm(
        range(iterations),
        desc=f"Computing sFID with {iterations} iterations",
        position=1,
        colour="#9E4545",
    ):
        tmp_dir.mkdir(exist_ok=False)  # User must provide an empty directory

        img_pair_list_copy = img_pair_list.copy()
        random.shuffle(img_pair_list_copy)
        half_size = len(img_pair_list) // 2
        split1 = img_pair_list_copy[:half_size]
        split2 = img_pair_list_copy[half_size:]

        real_split1 = Path(tmp_dir, "real1")
        real_split1.mkdir()
        real_split2 = Path(tmp_dir, "real2")
        real_split2.mkdir()
        gen_split1 = Path(tmp_dir, "gen1")
        gen_split1.mkdir()
        gen_split2 = Path(tmp_dir, "gen2")
        gen_split2.mkdir()

        for i, (real_path, gen_path) in enumerate(split1):
            ext = Path(real_path).suffix
            os.symlink(real_path, real_split1 / f"img_{i}{ext}")

            ext = Path(gen_path).suffix
            os.symlink(gen_path, gen_split1 / f"img_{i}{ext}")

        for i, (real_path, gen_path) in enumerate(split2):
            ext = Path(real_path).suffix
            os.symlink(real_path, real_split2 / f"img_{i}{ext}")

            ext = Path(gen_path).suffix
            os.symlink(gen_path, gen_split2 / f"img_{i}{ext}")

        split1_fid_value = calculate_fid_given_paths(
            [str(real_split1), str(gen_split2)], batch_size, device, dims, num_workers
        )

        split2_fid_value = calculate_fid_given_paths(
            [str(real_split2), str(gen_split1)], batch_size, device, dims, num_workers
        )

        fids.append(split1_fid_value)
        fids.append(split2_fid_value)

        shutil.rmtree(tmp_dir)

    sfid_value = sum(fids) / (iterations * 2)  # *2 since we do two splits
    return sfid_value
