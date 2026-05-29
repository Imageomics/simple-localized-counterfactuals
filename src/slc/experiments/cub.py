import random
import time
from argparse import ArgumentParser
from pathlib import Path

import torch
from filelock import FileLock
from PIL import Image
from saev.nn.modeling import MatryoshkaSparseAutoencoder, SparseAutoencoderConfig
from tqdm import tqdm

from slc.constants import (
    DatasetNames,
    get_cub_sae_model_path,
)
from slc.datasets import CUBDatasetInterface
from slc.experiments.configs.cfg_2 import cub_sae_configs
from slc.models import CUBDINOv3RecognitionModel, SDVAEPipeline
from slc.pipelines import (
    EditPipelineOutput,
    LatentEditPipeline,
    SAELatentEditPipeline,
)
from slc.utils import get_dataset_root, load_json, save_json


def run(
    save_dir: str,
    use_sae: bool,
    closest_class_dict: str,
    classifier_weights: str,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    cf_img_dir = save_dir / "imgs"
    cf_json_path = save_dir / "image_pair_list.json"
    label_info_path = save_dir / "label_info_list.json"
    flip_rate_results_path = save_dir / "flip_rate.json"
    lock_file = save_dir / "exp_lock.lock"

    classifier = CUBDINOv3RecognitionModel(class_num=200).cuda()
    classifier.eval()
    classifier.classifier.load_state_dict(torch.load(classifier_weights))
    vae = SDVAEPipeline()

    closest_class_dict = load_json(closest_class_dict)

    if use_sae:
        cfg = SparseAutoencoderConfig(
            d_vit=768,
            exp_factor=8,
            normalize_w_dec=True,
        )

        sae = MatryoshkaSparseAutoencoder(cfg=cfg).cuda()
        sae.eval()

        sae_path = get_cub_sae_model_path()
        sae.load_state_dict(torch.load(sae_path))

        edit_pipeline = SAELatentEditPipeline(
            recognition_model=classifier,
            ae_model=vae,
            configs=cub_sae_configs,
            sae=sae,
        )
    else:
        edit_pipeline = LatentEditPipeline(
            recognition_model=classifier, ae_model=vae, configs=cub_sae_configs
        )

    dset = CUBDatasetInterface(root=get_dataset_root(DatasetNames.CUB))
    dataset = dset.get_torch_dataset(phase="test", with_paths=True)

    cf_img_dir.mkdir(exist_ok=True, parents=True)
    image_pair_list = []

    flips = 0
    total = 0

    indicies = list(range(len(dataset)))
    random.seed(time.time())
    random.shuffle(indicies)

    tbar = tqdm(
        indicies,
        desc="CUB Experiment",
        colour="#3f618d",
        position=1,
    )
    for IDX in tbar:  # type: ignore
        total += 1
        out_path = cf_img_dir / f"{IDX}.png"

        if out_path.exists():
            continue

        image, label, p = dataset[IDX]

        tgt_label = closest_class_dict[str(label)]

        edit_output: EditPipelineOutput = edit_pipeline(
            image,
            src_label=label,
            tgt_label=tgt_label,
            is_binary=False,
        )
        edit_output.edit_image.resize(
            image.size, resample=Image.Resampling.BICUBIC
        ).save(out_path)
        image_pair_list.append([str(p), str(out_path.resolve())])
        save_json(image_pair_list, cf_json_path)
        with torch.no_grad():
            x = classifier.preprocess_image(image)
            out = classifier(x.unsqueeze(0).cuda())
            val, pred = torch.max(out, 1)
            org_pred = pred[0].item()

            x = classifier.preprocess_image(edit_output.edit_image)
            out = classifier(x.unsqueeze(0).cuda())
            val, pred = torch.max(out, 1)
            edit_pred = pred[0].item()

        with FileLock(lock_file) as lock:
            if label_info_path.exists():
                label_info_list = load_json(label_info_path)
            else:
                label_info_list = []

            label_info_list.append(
                {
                    "idx": IDX,
                    "src": label,
                    "tgt": tgt_label,
                    "org_pred": org_pred,
                    "edit_pred": edit_pred,
                }
            )

            save_json(label_info_list, label_info_path)

        if edit_pred == tgt_label:
            flips += 1

        save_json({"flip_rate": flips / total}, flip_rate_results_path)

        tbar.set_postfix({"FR": f"{flips / total:.4f}"})


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--classifier_weights", type=str, required=True)
    parser.add_argument("--closest_class_dict", type=str, required=True)
    parser.add_argument("--use_sae", action="store_true", default=False)
    args = parser.parse_args()

    run(
        args.save_dir,
        args.use_sae,
        args.closest_class_dict,
        args.classifier_weights,
    )
