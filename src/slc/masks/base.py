from abc import abstractmethod

import torch

from slc.models import BaseRecognitionModelInterface


class BaseMaskGenerator:
    def __init__(self, classifier: BaseRecognitionModelInterface, **kwargs):
        self.classifier: BaseRecognitionModelInterface = classifier

    @abstractmethod
    def create_masks(self, *args, **kwargs) -> torch.FloatTensor:
        raise NotImplementedError(
            f"create_masks method has not been implemented in {type(self).__name__}"
        )

    def _mask_from_saliency_map(
        self, saliency_map, threshold=0.8, only_positive=False, do_threshold=True
    ):
        if only_positive:
            saliency_map[saliency_map < 0] = 0
        saliency_map -= saliency_map.min()
        if saliency_map.max() != 0:
            saliency_map /= saliency_map.max()

        if do_threshold:
            vals, indicies = torch.flatten(saliency_map).sort()
            thresh_idx = max(0, int(len(vals) * threshold))
            thresh_idx = min(thresh_idx, len(vals) - 1)

            val_thresh = vals[thresh_idx]
            saliency_map[saliency_map >= val_thresh] = 1.0
            saliency_map[saliency_map < val_thresh] = 0.0

        return saliency_map
