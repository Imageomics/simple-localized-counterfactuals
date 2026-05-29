from argparse import ArgumentParser
from pathlib import Path

import torch
import torchvision.transforms.functional as F
from PIL import Image
from saev.nn.modeling import MatryoshkaSparseAutoencoder, SparseAutoencoderConfig
from tqdm import tqdm

from slc.constants import (
    DatasetNames,
    get_cheetah_cougar_label_map,
    get_egyptian_persian_label_map,
    get_imagenet_sae_model_path,
    get_zebra_sorrel_label_map,
)
from slc.datasets import ImageFolderDatasetInterface
from slc.experiments.configs.cfg_2 import configs, sae_configs
from slc.models import ResNet50PytorchRecognitionModel, SDVAEPipeline
from slc.pipelines import EditPipelineOutput, LatentEditPipeline, SAELatentEditPipeline
from slc.utils import get_dataset_root, save_json


def run(dset_name: DatasetNames, save_dir: str, use_sae: bool):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=False, parents=True)

    cf_img_dir = save_dir / "imgs"
    cf_json_path = save_dir / "image_pair_list.json"
    label_info_path = save_dir / "label_info_list.json"
    flip_rate_results_path = save_dir / "flip_rate.json"

    match dset_name:
        case DatasetNames.ZEBRA_SORREL:
            label_map = get_zebra_sorrel_label_map()
        case DatasetNames.CHEETAH_COUGAR:
            label_map = get_cheetah_cougar_label_map()
        case DatasetNames.EGYPTIAN_PERSIAN:
            label_map = get_egyptian_persian_label_map()
        case _:
            raise NotImplementedError(f"{dset_name} is not implemented!")

    classifier = ResNet50PytorchRecognitionModel(
        class_num=1000, use_imagenet_classifier=True
    )
    vae = SDVAEPipeline()

    if use_sae:
        print("Using SAE Edit Pipeline")
        cfg = SparseAutoencoderConfig(
            d_vit=2048,
            exp_factor=8,
            normalize_w_dec=True,
        )

        sae = MatryoshkaSparseAutoencoder(cfg=cfg).cuda()
        sae.eval()

        sae_path = get_imagenet_sae_model_path()
        sae.load_state_dict(torch.load(sae_path))

        edit_pipeline = SAELatentEditPipeline(
            recognition_model=classifier, ae_model=vae, configs=sae_configs, sae=sae
        )
    else:
        edit_pipeline = LatentEditPipeline(
            recognition_model=classifier, ae_model=vae, configs=configs
        )

    dset = ImageFolderDatasetInterface(root=get_dataset_root(dset_name))
    dataset = dset.get_torch_dataset(phase="train", with_paths=True)

    cf_img_dir.mkdir(exist_ok=True, parents=True)
    image_pair_list = []
    label_info_list = []

    flips = 0
    total = 0

    tbar = tqdm(
        enumerate(dataset),
        total=len(dataset),
        desc=f"{dset_name.value} Experiment",
        colour="#3f618d",
        position=1,
    )
    for IDX, (image, label, p) in tbar:  # type: ignore
        total += 1
        image_net_src_label = label_map[label]
        if label == 0:
            image_net_tgt_label = label_map[1]
        elif label == 1:
            image_net_tgt_label = label_map[0]

        edit_output: EditPipelineOutput = edit_pipeline(
            image, src_label=image_net_src_label, tgt_label=image_net_tgt_label
        )
        out_path = cf_img_dir / f"{IDX}.png"
        edit_output.edit_image.resize(
            image.size, resample=Image.Resampling.BICUBIC
        ).save(out_path)
        image_pair_list.append([str(p), str(out_path.resolve())])
        save_json(image_pair_list, cf_json_path)
        with torch.no_grad():
            x = classifier.preprocess_tensor_image(F.to_tensor(image))
            out = classifier(x.unsqueeze(0).cuda())
            val, pred = torch.max(out, 1)
            org_pred = pred[0].item()

            x = classifier.preprocess_tensor_image(F.to_tensor(edit_output.edit_image))
            out = classifier(x.unsqueeze(0).cuda())
            val, pred = torch.max(out, 1)
            edit_pred = pred[0].item()

        label_info_list.append(
            {
                "src": image_net_src_label,
                "tgt": image_net_tgt_label,
                "org_pred": org_pred,
                "edit_pred": edit_pred,
            }
        )

        save_json(label_info_list, label_info_path)

        if edit_pred == image_net_tgt_label:
            flips += 1

        save_json({"flip_rate": flips / total}, flip_rate_results_path)

        tbar.set_postfix({"FR": f"{flips / total:.4f}"})


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["zebra_sorrel", "cheetah_cougar", "egyptian_persian"],
        default="zebra_sorrel",
    )
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--use_sae", action="store_true", default=False)
    args = parser.parse_args()

    match args.dataset:
        case "zebra_sorrel":
            dset_name = DatasetNames.ZEBRA_SORREL
        case "cheetah_cougar":
            dset_name = DatasetNames.CHEETAH_COUGAR
        case "egyptian_persian":
            dset_name = DatasetNames.EGYPTIAN_PERSIAN
        case _:
            raise NotImplementedError(
                f"{args.dataset} is not implemented for this script!"
            )

    run(dset_name, args.save_dir, args.use_sae)
