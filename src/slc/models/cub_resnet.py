from typing import Any

import timm
import torch
import torchvision.transforms.functional as F
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from .base import (
    BaseFeatureExtractorInterface,
    BaseRecognitionModelInterface,
)


class CUBResNetFeatureExtractor(BaseFeatureExtractorInterface):
    def __init__(self):
        super().__init__()
        self.hf_path = "hf_hub:anonauthors/cub200-resnet50"
        self.full_model: torch.nn.Module = timm.create_model(
            self.hf_path, pretrained=True
        )

        self.image_processor = create_transform(
            **resolve_data_config(self.full_model.pretrained_cfg, model=self.full_model)
        )

        self.hidden_size = 2048

    def preprocess_image(self, img, **kwargs) -> Any:
        x: torch.Tensor = self.image_processor(img)  # type: ignore
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        return x

    def get_normalize_values(self) -> dict[str, list[float]]:
        return {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }

    def preprocess_tensor_image(self, img: torch.Tensor, **kwargs):
        img = F.resize(
            img,
            size=[224, 224],
            interpolation=F.InterpolationMode.BICUBIC,
            antialias=True,
        )
        img = F.normalize(img, **self.get_normalize_values())  # type: ignore
        return img

    def forward(self, x, **kwargs) -> Any:
        feats = self.full_model.forward_features(x)
        feats_pooled = self.full_model.forward_head(feats, pre_logits=True)
        return feats_pooled

    def forward_spatial(self, x, **kwargs) -> Any:
        outputs = self.full_model.forward_features(x)
        return outputs

    def forward_spatial_to_pooled(self, x, **kwargs) -> Any:
        feats_pooled = self.full_model.forward_head(x, pre_logits=True)
        return feats_pooled


class CUBResNet50RecognitionModel(BaseRecognitionModelInterface):
    def __init__(self):
        feature_extractor = CUBResNetFeatureExtractor()
        classifier = feature_extractor.full_model.get_classifier()
        super().__init__(feature_extractor=feature_extractor, classifier=classifier)
