import torch
from captum.attr import IntegratedGradients

from slc.models.base import BaseRecognitionModelInterface

from .base import BaseMaskGenerator


class IntegratedGradientsMaskGenerator(BaseMaskGenerator):
    def __init__(self, classifier: BaseRecognitionModelInterface, **kwargs):
        super().__init__(classifier, **kwargs)
        self.saliency = IntegratedGradients(self.classifier)

    def create_masks(
        self,
        x,
        target,
        baselines=None,
        threshold=0.8,
        only_positive=False,
        do_threshold=True,
    ) -> torch.FloatTensor:
        saliency_map = self.saliency.attribute(x, target=target, baselines=baselines)
        saliency_map = saliency_map.detach().cpu()
        saliency_map = saliency_map[0].mean(0)
        saliency_map = self._mask_from_saliency_map(
            saliency_map, threshold, only_positive, do_threshold
        )

        return saliency_map
