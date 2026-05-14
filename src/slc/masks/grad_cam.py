from abc import abstractmethod
from typing import Any

import torch
from pytorch_grad_cam import FinerCAM, GradCAM, GradCAMPlusPlus, LayerCAM, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from slc.models import (
    BaseRecognitionModelInterface,
    CLIPRecognitionModel,
    DINOv2RecognitionModel,
    DINOv3RecognitionModel,
)

from .base import BaseMaskGenerator


class GradCAMBaseMaskGenerator(BaseMaskGenerator):
    def __init__(
        self,
        classifier: BaseRecognitionModelInterface,
        target_layers: Any = None,
        reshape_transform_fn: Any = None,
        **kwargs,
    ):
        super().__init__(classifier, **kwargs)
        if reshape_transform_fn is None:
            reshape_transform_fn = self._get_default_reshape_transform()
        self._set_saliency(target_layers, reshape_transform_fn)

    @abstractmethod
    def _set_saliency(self, target_layers, reshape_transform_fn):
        raise NotImplementedError(
            f"_set_saliency method not implemnted in {type(self).__name__}"
        )

    def _get_default_reshape_transform(self):
        pipeline_type = type(self.classifier)
        if pipeline_type in [
            CLIPRecognitionModel,
            DINOv2RecognitionModel,
            DINOv3RecognitionModel,
        ]:

            def reshape_transform(tensor, height=7, width=7):
                result = tensor[:, 1:, :].reshape(
                    tensor.size(0), height, width, tensor.size(2)
                )

                # Bring the channels to the first dimension,
                # like in CNNs.
                result = result.transpose(2, 3).transpose(1, 2)
                return result

            return reshape_transform
        else:
            return None

    def create_masks(
        self,
        x,
        target,
        threshold=0.8,
        only_positive=False,
        do_threshold=True,
    ) -> torch.FloatTensor:
        targets = [ClassifierOutputTarget(target)]
        saliency_map = self.saliency(x, targets=targets)
        saliency_map = torch.from_numpy(saliency_map)
        saliency_map = saliency_map[0]
        saliency_map = self._mask_from_saliency_map(
            saliency_map, threshold, only_positive, do_threshold
        )

        return saliency_map


class GradCAMMaskGenerator(GradCAMBaseMaskGenerator):
    def _set_saliency(self, target_layers, reshape_transform_fn):
        self.saliency = GradCAM(
            model=self.classifier,
            target_layers=target_layers,
            reshape_transform=reshape_transform_fn,
        )


class GradCAMPPMaskGenerator(GradCAMBaseMaskGenerator):
    def _set_saliency(self, target_layers, reshape_transform_fn):
        self.saliency = GradCAMPlusPlus(
            model=self.classifier,
            target_layers=target_layers,
            reshape_transform=reshape_transform_fn,
        )


class LayerCAMMaskGenerator(GradCAMBaseMaskGenerator):
    def _set_saliency(self, target_layers, reshape_transform_fn):
        self.saliency = LayerCAM(
            model=self.classifier,
            target_layers=target_layers,
            reshape_transform=reshape_transform_fn,
        )


class ScoreCAMMaskGenerator(GradCAMBaseMaskGenerator):
    def _set_saliency(self, target_layers, reshape_transform_fn):
        self.saliency = ScoreCAM(
            model=self.classifier,
            target_layers=target_layers,
            reshape_transform=reshape_transform_fn,
        )


class FinerCAMMaskGenerator(GradCAMBaseMaskGenerator):
    def _set_saliency(self, target_layers, reshape_transform_fn):
        self.saliency = FinerCAM(
            model=self.classifier,
            target_layers=target_layers,
            reshape_transform=reshape_transform_fn,
        )

    def create_masks(  # type: ignore
        self,
        x,
        target,
        comparison_targets,
        eigen_smooth=False,
        alpha=1,
        threshold=0.8,
        only_positive=False,
        do_threshold=True,
    ) -> torch.FloatTensor:
        saliency_map = self.saliency(
            x,
            targets=None,
            target_idx=target,
            comparison_categories=comparison_targets,
            eigen_smooth=eigen_smooth,
            alpha=alpha,
        )
        saliency_map = torch.from_numpy(saliency_map)
        saliency_map = saliency_map[0]
        saliency_map = self._mask_from_saliency_map(
            saliency_map, threshold, only_positive, do_threshold
        )

        return saliency_map
