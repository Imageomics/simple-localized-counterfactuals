from argparse import ArgumentParser

from slc.utils import load_json


def run(img_label_list: str) -> float:
    img_label_list = load_json(img_label_list)

    total = len(img_label_list)
    correct = 0
    for item in img_label_list:
        if item["edit_pred"] == item["tgt"]:
            correct += 1
    flip_rate = correct / total
    return flip_rate


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--img_label_list", type=str)
    args = parser.parse_args()

    fr = run(args.img_label_list)

    print(f"Flip rate score based on {args.img_label_list}: {fr}")
