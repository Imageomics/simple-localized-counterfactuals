import random
import time
from argparse import ArgumentParser
from pathlib import Path

import torch
import torchvision.transforms.functional as F
from filelock import FileLock
from PIL import Image
from saev.nn.modeling import MatryoshkaSparseAutoencoder, SparseAutoencoderConfig
from tqdm import tqdm

from slc.constants import (
    DatasetNames,
    get_celeba_hq_sae_model_path,
    get_celeba_sae_model_path,
)
from slc.datasets import CelebADatasetInterface, CelebAHQDatasetInterface
from slc.experiments.configs.cfg_2 import celeba_configs, sae_celeba_configs
from slc.models import CelebAHQRecognitionModel, CelebARecognitionModel, SDVAEPipeline
from slc.pipelines import (
    EditPipelineOutput,
    LatentEditPipeline,
    SAELatentEditPipeline,
)
from slc.utils import get_dataset_root, load_json, save_json


def run(
    dset_name: DatasetNames,
    save_dir: str,
    use_sae: bool,
    query_label: int,
    classifier_weights: str,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    cf_img_dir = save_dir / "imgs"
    cf_json_path = save_dir / "image_pair_list.json"
    label_info_path = save_dir / "label_info_list.json"
    flip_rate_results_path = save_dir / "flip_rate.json"
    lock_file = save_dir / "exp_lock.lock"

    if dset_name == DatasetNames.CELEBA:
        classifier = CelebARecognitionModel(weight_path=classifier_weights)
    elif dset_name == DatasetNames.CELEBA_HQ:
        classifier = CelebAHQRecognitionModel(weight_path=classifier_weights)
    vae = SDVAEPipeline()

    if use_sae:
        cfg = SparseAutoencoderConfig(
            d_vit=1024,
            exp_factor=8,
            normalize_w_dec=True,
        )

        sae = MatryoshkaSparseAutoencoder(cfg=cfg).cuda()
        sae.eval()

        if dset_name == DatasetNames.CELEBA:
            sae_path = get_celeba_sae_model_path()
        elif dset_name == DatasetNames.CELEBA_HQ:
            sae_path = get_celeba_hq_sae_model_path()
        sae.load_state_dict(torch.load(sae_path))

        edit_pipeline = SAELatentEditPipeline(
            recognition_model=classifier,
            ae_model=vae,
            configs=sae_celeba_configs,
            sae=sae,
        )
    else:
        edit_pipeline = LatentEditPipeline(
            recognition_model=classifier, ae_model=vae, configs=celeba_configs
        )

    if dset_name == DatasetNames.CELEBA:
        dset = CelebADatasetInterface(root=get_dataset_root(dset_name))
    elif dset_name == DatasetNames.CELEBA_HQ:
        dset = CelebAHQDatasetInterface(root=get_dataset_root(dset_name))

    dataset = dset.get_torch_dataset(
        phase="val", query_label=query_label, with_paths=True
    )

    cf_img_dir.mkdir(exist_ok=True, parents=True)
    image_pair_list = []

    flips = 0
    total = 0

    indicies = list(range(len(dataset)))
    random.seed(time.time())
    random.shuffle(indicies)

    tbar = tqdm(
        indicies,
        desc=f"{dset_name.value} Experiment",
        colour="#3f618d",
        position=1,
    )
    for IDX in tbar:  # type: ignore
        total += 1
        out_path = cf_img_dir / f"{IDX}.png"

        if out_path.exists():
            continue

        image, label, p = dataset[IDX]

        if dset_name == DatasetNames.CELEBA_HQ:
            ql = 0  # gender
            if query_label in [31, 39]:
                ql = 1 if query_label == 31 else 2
                # 1 -> smile, 2 -> age
        else:
            ql = query_label

        edit_output: EditPipelineOutput = edit_pipeline(
            image,
            src_label=label,
            tgt_label=ql,
            is_binary=True,
        )
        edit_output.edit_image.resize(
            image.size, resample=Image.Resampling.BICUBIC
        ).save(out_path)
        image_pair_list.append([str(p), str(out_path.resolve())])
        save_json(image_pair_list, cf_json_path)
        with torch.no_grad():
            x = classifier.preprocess_tensor_image(F.to_tensor(image))
            out = classifier(x.unsqueeze(0).cuda())

            pred = torch.round(torch.nn.functional.sigmoid(out[:, ql]))
            org_pred = pred.item()

            x = classifier.preprocess_tensor_image(F.to_tensor(edit_output.edit_image))
            out = classifier(x.unsqueeze(0).cuda())
            pred = torch.round(torch.nn.functional.sigmoid(out[:, ql]))
            edit_pred = pred.item()

        with FileLock(lock_file) as lock:
            if label_info_path.exists():
                label_info_list = load_json(label_info_path)
            else:
                label_info_list = []

            label_info_list.append(
                {
                    "idx": IDX,
                    "src": label,
                    "tgt": 1 - label,
                    "query_label": ql,
                    "org_pred": org_pred,
                    "edit_pred": edit_pred,
                }
            )

            save_json(label_info_list, label_info_path)

        if edit_pred == (1 - label):
            flips += 1

        save_json({"flip_rate": flips / total}, flip_rate_results_path)

        tbar.set_postfix({"FR": f"{flips / total:.4f}"})


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["celeba", "celeba-hq"],
        default="celeba",
    )
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--classifier_weights", type=str, required=True)
    parser.add_argument("--query_label", type=int, required=True, default=31)
    parser.add_argument("--use_sae", action="store_true", default=False)
    args = parser.parse_args()

    match args.dataset:
        case "celeba":
            dset_name = DatasetNames.CELEBA
        case "celeba-hq":
            dset_name = DatasetNames.CELEBA_HQ
        case _:
            raise NotImplementedError(
                f"{args.dataset} is not implemented for this script!"
            )

    run(
        dset_name,
        args.save_dir,
        args.use_sae,
        args.query_label,
        args.classifier_weights,
    )
