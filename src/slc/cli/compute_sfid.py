from argparse import ArgumentParser

from slc.evaluation.fid import compute_sfid
from slc.utils import load_json


def run(
    tmp_dir: str,
    img_pair_list: str,
    num_workers: int = 4,
    batch_size: int = 32,
    iterations: int = 10,
) -> float:
    img_pair_list = load_json(img_pair_list)

    sfid = compute_sfid(
        img_pair_list=img_pair_list,
        tmp_dir=tmp_dir,
        iterations=iterations,
        num_workers=num_workers,
        batch_size=batch_size,
    )

    return sfid


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--img_pair_list", type=str)
    parser.add_argument("--tmp_dir", type=str)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    sfid = run(
        args.tmp_dir,
        args.img_pair_list,
        args.num_workers,
        batch_size=args.batch_size,
        iterations=args.iterations,
    )

    print(f"sFID score based on {args.img_pair_list}: {sfid}")
