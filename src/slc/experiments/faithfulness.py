import random
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np
import quantus
import torch
from tqdm import tqdm

from slc.constants import (
    DatasetNames,
    get_cheetah_cougar_label_map,
    get_egyptian_persian_label_map,
    get_zebra_sorrel_label_map,
)
from slc.datasets import CUBDatasetInterface, ImageFolderDatasetInterface
from slc.experiments.configs.cfg_2 import configs
from slc.masks import (
    AdversarialGradientIntegrationMaskGenerator,
    DeepLiftMaskGenerator,
    GradCAMMaskGenerator,
    GuidedBackPropagationMaskGenerator,
    GuidedIntegratedGradientsMaskGenerator,
    InputXGradientMaskGenerator,
    IntegratedGradientsMaskGenerator,
    LayerCAMMaskGenerator,
    LimeMaskGenerator,
    LRPMaskGenerator,
    OcclusionMaskGenerator,
    ScoreCAMMaskGenerator,
)
from slc.masks.lag import LatentEditMaskGenerator
from slc.models import (
    CUBDINOv3RecognitionModel,
    ResNet50PytorchRecognitionModel,
    SDVAEPipeline,
)
from slc.pipelines import (
    LatentEditPipeline,
)
from slc.utils import get_dataset_root, load_json, save_json


def run(
    dset_name: DatasetNames,
    save_dir: str,
    ablation: str = "latent_mask",
    seed: int = 42,
    sample_size: int = 100,
    per_class_cub: int = 5,
    cub_closet_class_dict: str = "",
    classifier_weights: str = "",
):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    match dset_name:
        case DatasetNames.ZEBRA_SORREL:
            label_map = get_zebra_sorrel_label_map()
        case DatasetNames.CHEETAH_COUGAR:
            label_map = get_cheetah_cougar_label_map()
        case DatasetNames.EGYPTIAN_PERSIAN:
            label_map = get_egyptian_persian_label_map()
        case DatasetNames.CUB:
            closest_class_dict = load_json(cub_closet_class_dict)
            label_map = closest_class_dict
        case _:
            raise NotImplementedError(f"{dset_name} is not implemented!")

    gc_reshape_transform = None

    if dset_name == DatasetNames.CUB:
        classifier = CUBDINOv3RecognitionModel(class_num=200).cuda()
        classifier.classifier.load_state_dict(torch.load(classifier_weights))
        grad_cam_layers = [classifier.feature_extractor.image_encoder.layer[-1].norm1]

        def dino_reshape_transform(tensor, height=14, width=14):
            num_registers = (
                classifier.feature_extractor.image_encoder.config.num_register_tokens
            )
            result = tensor[:, 1 + num_registers :, :].reshape(
                tensor.size(0), height, width, tensor.size(2)
            )
            result = result.transpose(2, 3).transpose(1, 2)
            return result

        gc_reshape_transform = dino_reshape_transform
    else:
        classifier = ResNet50PytorchRecognitionModel(
            class_num=1000, use_imagenet_classifier=True
        ).cuda()
        grad_cam_layers = [classifier.feature_extractor.image_encoder.layer4[-1]]
    classifier.eval()

    pixel_flipping = quantus.PixelFlipping(  # The lower the better
        features_in_step=224,
        perturb_baseline="black",
        perturb_func=quantus.perturb_func.batch_baseline_replacement_by_indices,
        return_auc_per_sample=True,
    )

    faithful_correlation = quantus.FaithfulnessCorrelation(  # Higher the better
        nr_runs=100,
        subset_size=224,
        perturb_baseline="black",
        perturb_func=quantus.perturb_func.batch_baseline_replacement_by_indices,
        similarity_func=quantus.similarity_func.correlation_pearson,
        abs=True,
    )

    match ablation:
        case "lag":
            vae = SDVAEPipeline()
            configs.max_iter_steps = 1
            edit_pipeline = LatentEditPipeline(
                recognition_model=classifier, ae_model=vae, configs=configs
            )
            mask_generator = LatentEditMaskGenerator(classifier, edit_pipeline)

        case "integrated_gradients":
            mask_generator = IntegratedGradientsMaskGenerator(classifier)
        case "gig":
            mask_generator = GuidedIntegratedGradientsMaskGenerator(classifier)
        case "agi":
            mask_generator = AdversarialGradientIntegrationMaskGenerator(classifier)
        case "deep_lift":
            mask_generator = DeepLiftMaskGenerator(classifier)
        case "gbp":
            mask_generator = GuidedBackPropagationMaskGenerator(classifier)
        case "grad_cam":
            mask_generator = GradCAMMaskGenerator(
                classifier,
                target_layers=grad_cam_layers,
                reshape_transform_fn=gc_reshape_transform,
            )
        case "layer_cam":
            mask_generator = LayerCAMMaskGenerator(
                classifier,
                target_layers=grad_cam_layers,
                reshape_transform_fn=gc_reshape_transform,
            )
        case "score_cam":
            mask_generator = ScoreCAMMaskGenerator(
                classifier,
                target_layers=grad_cam_layers,
                reshape_transform_fn=gc_reshape_transform,
            )
        case "ixg":
            mask_generator = InputXGradientMaskGenerator(classifier)
        case "lrp":
            mask_generator = LRPMaskGenerator(classifier)
        case "occlusion":
            mask_generator = OcclusionMaskGenerator(classifier)
        case "lime":
            mask_generator = LimeMaskGenerator(classifier)
        case _:
            raise NotImplementedError(f"Ablation {ablation} not implemented!")

    if dset_name == DatasetNames.CUB:
        dset = CUBDatasetInterface(root=get_dataset_root(dset_name))
        dataset = dset.get_torch_dataset(phase="test", with_paths=True)
    else:
        dset = ImageFolderDatasetInterface(root=get_dataset_root(dset_name))
        dataset = dset.get_torch_dataset(phase="train", with_paths=True)

    indices = list(range(len(dataset)))
    random.seed(seed)
    random.shuffle(indices)

    if dset_name == DatasetNames.CUB:
        filtered_indicies = []
        class_count = defaultdict(int)
        for idx in indices:
            _, lbl, _ = dataset[idx]
            if class_count[lbl] >= per_class_cub:
                continue

            filtered_indicies.append(idx)
            class_count[lbl] += 1
        indices = filtered_indicies
    else:
        indices = indices[:sample_size]

    all_scores = {
        "pixel_flipping": [],
        "faithfulness_correlation": [],
    }

    device = torch.device("cuda:0")

    tbar = tqdm(indices, "Evaluating Faithfulness (Pixel Flipping)")
    for idx in tbar:
        imgs, lbls, paths = dataset[idx]

        if dset_name == DatasetNames.CUB:
            image_net_src_label = lbls
            image_net_tgt_label = label_map[str(lbls)]
        else:
            image_net_src_label = label_map[lbls]
            if lbls == 0:
                image_net_tgt_label = label_map[1]
            elif lbls == 1:
                image_net_tgt_label = label_map[0]

        tensor_input = classifier.preprocess_image(imgs).unsqueeze(0).float()
        if ablation == "lag":
            a_batch = mask_generator.create_masks(  # type: ignore
                imgs,
                source_label=image_net_src_label,  # type: ignore
                target_label=image_net_tgt_label,  # type: ignore
                do_threshold=False,
            ).unsqueeze(0)
        else:
            a_batch = mask_generator.create_masks(  # type: ignore
                tensor_input.cuda(),
                target=image_net_src_label,  # type: ignore
                do_threshold=False,
            ).unsqueeze(0)

        a_batch = a_batch.cpu().numpy()
        x_batch = tensor_input.cpu().numpy()

        y_batch = torch.tensor(lbls).unsqueeze(0).cpu().numpy()
        scores = pixel_flipping(
            model=classifier,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            device=device,
        )

        all_scores["pixel_flipping"].extend(scores)

        scores = faithful_correlation(
            model=classifier,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            device=device,
        )

        all_scores["faithfulness_correlation"].extend(scores)

        tbar.set_postfix(
            {
                "pixel_flipping": np.array(all_scores["pixel_flipping"]).mean(),
                "faithfulness_correlation": np.array(
                    all_scores["faithfulness_correlation"]
                ).mean(),
            }
        )
        save_json(all_scores, save_dir / f"{ablation}_scores.json")
    print(np.array(all_scores["pixel_flipping"]).mean())
    print(np.array(all_scores["faithfulness_correlation"]).mean())

    save_json(all_scores, save_dir / f"{ablation}_scores.json")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["zebra_sorrel", "cheetah_cougar", "egyptian_persian", "cub"],
        default="zebra_sorrel",
    )
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument(
        "--ablation",
        type=str,
        choices=[
            "integrated_gradients",
            "lag",
            "gig",
            "agi",
            "deep_lift",
            "gbp",
            "grad_cam",
            "layer_cam",
            "score_cam",
            "ixg",
            "lrp",
            "occlusion",
            "lime",
        ],
        default="lag",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--per_class_cub", type=int, default=5)
    parser.add_argument("--cub_closest_class_dict", type=str, default="")
    parser.add_argument("--classifier_weights", type=str, default="")
    args = parser.parse_args()

    match args.dataset:
        case "zebra_sorrel":
            dset_name = DatasetNames.ZEBRA_SORREL
        case "cheetah_cougar":
            dset_name = DatasetNames.CHEETAH_COUGAR
        case "egyptian_persian":
            dset_name = DatasetNames.EGYPTIAN_PERSIAN
        case "cub":
            dset_name = DatasetNames.CUB
        case _:
            raise NotImplementedError(
                f"{args.dataset} is not implemented for this script!"
            )

    run(
        dset_name,
        args.save_dir,
        args.ablation,
        args.seed,
        args.sample_size,
        args.per_class_cub,
        args.cub_closest_class_dict,
        args.classifier_weights,
    )
