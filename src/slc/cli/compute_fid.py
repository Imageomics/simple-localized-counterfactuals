import os
import shutil
from argparse import ArgumentParser
from pathlib import Path

from slc.evaluation.fid import compute_fid
from slc.utils import load_json


def run(
    tmp_dir: str, img_pair_list: str, num_workers: int = 4, batch_size: int = 32
) -> float:
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(exist_ok=False)  # User must provide an empty directory
    real_dir = tmp_dir / "real"
    gen_dir = tmp_dir / "gen"
    real_dir.mkdir()
    gen_dir.mkdir()

    img_pair_list = load_json(img_pair_list)

    for i, (real_path, gen_path) in enumerate(img_pair_list):
        ext = Path(real_path).suffix
        os.symlink(real_path, real_dir / f"img_{i}{ext}")

        ext = Path(gen_path).suffix
        os.symlink(gen_path, gen_dir / f"img_{i}{ext}")

    fid = compute_fid(str(real_dir), str(gen_dir), num_workers, batch_size=batch_size)

    shutil.rmtree(tmp_dir)

    return fid


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--img_pair_list", type=str)
    parser.add_argument("--tmp_dir", type=str)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    fid = run(
        args.tmp_dir, args.img_pair_list, args.num_workers, batch_size=args.batch_size
    )

    print(f"FID score based on {args.img_pair_list}: {fid}")
