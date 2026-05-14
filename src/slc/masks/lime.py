import torch
from captum.attr import Lime

from slc.models.base import BaseRecognitionModelInterface

from .base import BaseMaskGenerator


class LimeMaskGenerator(BaseMaskGenerator):
    def __init__(self, classifier: BaseRecognitionModelInterface, **kwargs):
        super().__init__(classifier, **kwargs)
        self.saliency = Lime(self.classifier)

    def create_masks(
        self,
        x,
        target,
        n_samples=200,
        threshold=0.8,
        only_positive=False,
        do_threshold=True,
    ) -> torch.FloatTensor:
        saliency_map = self.saliency.attribute(x, target=target, n_samples=n_samples)
        saliency_map = saliency_map.detach().cpu()
        saliency_map = saliency_map[0].mean(0)
        saliency_map = self._mask_from_saliency_map(
            saliency_map, threshold, only_positive, do_threshold
        )

        return saliency_map
