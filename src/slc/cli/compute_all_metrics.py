import shutil
from argparse import ArgumentParser
from pathlib import Path

from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from slc.utils import load_json, save_json

from .compute_cout import is_binary_dataset
from .compute_cout import run as compute_cout
from .compute_fid import run as compute_fid
from .compute_flip_rate import run as compute_flip_rate
from .compute_sfid import run as compute_sfid
from .compute_sss import run as compute_sss


def handle_celeba_hq(tmp_dir, img_pair_list):
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(exist_ok=False)

    img_dir = tmp_dir / "imgs"
    img_dir.mkdir()
    new_tmp_dir = tmp_dir / "new_tmp"

    data = load_json(img_pair_list)
    for i, (org_path, edit_path) in tqdm(
        enumerate(data), total=len(data), desc="copying CelebA-HQ images"
    ):
        org_path = Path(org_path)
        parts = list(org_path.parts)
        parts[-2] = "PngImg"
        parts[-1] = f"{org_path.parts[-1].split('.')[0]}.png"
        data[i][0] = str("/".join(parts))
        print(data[i][0])

    new_img_pair_list = tmp_dir / "image_pair_list.json"
    save_json(data, new_img_pair_list)
    return str(new_tmp_dir), str(new_img_pair_list)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--img_pair_list", type=str)
    parser.add_argument("--img_label_list", type=str)
    parser.add_argument("--query_label", type=int, default=0)
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="celeba-hq")
    parser.add_argument(
        "--cout_model_weights",
        type=str,
        default="/home/carlyn.1/code/rcvcf/weights/celeba_classifier.pth",
    )
    parser.add_argument(
        "--sss_model_weights", type=str, default="weights/checkpoint_0099.pth.tar"
    )
    parser.add_argument("--tmp_dir", type=str)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--cout_batch_size", type=int, default=32)
    parser.add_argument("--sfid_iterations", type=int, default=10)
    parser.add_argument("--only_one", type=str, default="none")
    args = parser.parse_args()

    if args.dataset == "celeba-hq":
        new_tmp_dir, new_img_pair_list = handle_celeba_hq(
            args.tmp_dir, args.img_pair_list
        )

        org_tmp_dir = args.tmp_dir
        args.tmp_dir = new_tmp_dir
        args.img_pair_list = new_img_pair_list

    fid = 0
    if args.only_one in ["none", "fid"]:
        fid = compute_fid(
            args.tmp_dir,
            args.img_pair_list,
            args.num_workers,
            batch_size=args.batch_size,
        )

    print(f"FID score based on {args.img_pair_list}: {fid}")

    sfid = 0
    if args.only_one in ["none", "sfid"]:
        sfid = compute_sfid(
            args.tmp_dir,
            args.img_pair_list,
            args.num_workers,
            batch_size=args.batch_size,
            iterations=args.sfid_iterations,
        )

    print(f"sFID score based on {args.img_pair_list}: {sfid}")

    cout = 0
    if args.only_one in ["none", "cout"]:
        cout = compute_cout(
            args.img_label_list,
            args.img_pair_list,
            args.query_label,
            args.target_label,
            args.dataset,
            args.cout_model_weights,
            args.num_workers,
            args.cout_batch_size,
        )

        if not is_binary_dataset(args.dataset):
            cout2 = compute_cout(
                args.img_label_list,
                args.img_pair_list,
                args.target_label,  # Switch with target
                args.query_label,  # Switch with query
                args.dataset,
                args.cout_model_weights,
                args.num_workers,
                args.batch_size,
            )

            cout = (cout + cout2) / 2.0  # Average the two

    print(f"COUT score based on {args.img_pair_list}: {cout}")

    sss = 0
    if args.only_one in ["none", "sss"]:
        sss = compute_sss(
            args.img_pair_list,
            args.sss_model_weights,
            args.num_workers,
            args.batch_size,
        )

    print(f"SSS score based on {args.img_pair_list}: {sss:>4f}")

    fr = 0
    if args.only_one in ["none", "fr"]:
        fr = compute_flip_rate(args.img_label_list)

    print(f"Flip rate score based on {args.img_label_list}: {fr}")

    # Display results
    console = Console()
    table = Table(
        title=f"Metrics for {args.img_pair_list}",
        show_header=True,
        header_style="bold magenta",
    )

    table.add_column("FID", style="green", justify="center")
    table.add_column("sFID", style="cyan", justify="center")
    table.add_column("SSS", style="magenta", justify="center")
    table.add_column("COUT", style="yellow", justify="center")
    table.add_column("FR", style="red", justify="center")

    table.add_row(f"{fid:.4f}", f"{sfid:.4f}", f"{sss:.4f}", f"{cout:.4f}", f"{fr:.4f}")

    console.print(table)

    if args.dataset == "celeba-hq":
        shutil.rmtree(org_tmp_dir)
