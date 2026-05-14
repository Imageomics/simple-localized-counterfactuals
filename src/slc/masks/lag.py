import torch
import torchvision.transforms.functional as F

from slc.models import BaseRecognitionModelInterface
from slc.pipelines import LatentEditPipeline

from .base import BaseMaskGenerator


class LatentEditMaskGenerator(BaseMaskGenerator):
    def __init__(
        self,
        classifier: BaseRecognitionModelInterface,
        edit_pipeline: LatentEditPipeline,
        **kwargs,
    ):
        super().__init__(classifier, **kwargs)
        self.edit_pipeline: LatentEditPipeline = edit_pipeline

    def create_masks(
        self,
        x,
        source_label,
        target_label,
        threshold=0.8,
        only_positive=False,
        do_threshold=True,
    ) -> torch.FloatTensor:
        edit_output = self.edit_pipeline(
            x, src_label=source_label, tgt_label=target_label
        )
        processed_img = self.classifier.preprocess_image(x)
        # saliency_map = self.saliency.attribute(x, target=target, baselines=baselines)
        saliency_map = edit_output.full_mask
        saliency_map = saliency_map.detach().cpu()
        saliency_map = saliency_map.mean(1)
        saliency_map = F.resize(
            saliency_map,
            (processed_img.shape[2:]),
            interpolation=F.InterpolationMode.BICUBIC,
        )
        saliency_map = saliency_map[0]

        saliency_map = self._mask_from_saliency_map(
            saliency_map, threshold, only_positive, do_threshold
        )

        return saliency_map
