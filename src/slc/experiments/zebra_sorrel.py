from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from slc.constants import DatasetNames
from slc.datasets import ImageFolderDatasetInterface
from slc.experiments.configs.cfg_1 import configs
from slc.models import ResNet50RecognitionModel, SDVAEPipeline
from slc.pipelines import EditPipelineOutput, LatentEditPipeline
from slc.utils import get_dataset_root, save_json


def run():
    classifier = ResNet50RecognitionModel(class_num=1000, use_imagenet_classifier=True)
    vae = SDVAEPipeline()

    edit_pipeline = LatentEditPipeline(
        recognition_model=classifier, ae_model=vae, configs=configs
    )

    dset = ImageFolderDatasetInterface(root=get_dataset_root(DatasetNames.ZEBRA_SORREL))
    zebra_sorrel_dataset = dset.get_torch_dataset(phase="train", with_paths=True)
    zebra_sorrel_label_map = {
        0: 339,  # sorrel
        1: 340,  # zebra
    }

    tmp_dir = Path("/local/scratch2/carlyn.1/rcvcf/tmp/")
    cf_dir = tmp_dir / "tmp_zs_results"
    cf_img_dir = cf_dir / "imgs"
    cf_json_path = cf_dir / "image_pair_list.json"
    label_info_path = cf_dir / "label_info_list.json"

    cf_img_dir.mkdir(exist_ok=True, parents=True)
    image_pair_list = []
    label_info_list = []
    for IDX, (image, label, p) in tqdm(
        enumerate(zebra_sorrel_dataset),
        total=len(zebra_sorrel_dataset),
        desc="Zebra-Sorrel Experiment",
        colour="#3f618d",
        position=1,
    ):  # type: ignore
        image_net_src_label = zebra_sorrel_label_map[label]
        if label == 0:
            image_net_tgt_label = zebra_sorrel_label_map[1]
        elif label == 1:
            image_net_tgt_label = zebra_sorrel_label_map[0]

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
            x = classifier.preprocess_image(image)
            out = classifier(x.unsqueeze(0).cuda())
            val, pred = torch.max(out, 1)
            org_pred = pred[0].item()

            x = classifier.preprocess_image(edit_output.edit_image)
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

    results_dir = Path("/local/scratch2/carlyn.1/rcvcf/results")
    results_dir.mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    run()
