from typing import Any

import torch
import torchvision.transforms.functional as F
from PIL import Image

from .base import BaseRecognitionModelInterface, BasicLinearClassifier
from .dino import (
    DINOFeatureExtractionType,
    DINOv3FeatureExtractor,
)


class CUBDINOv3FeatureExtractor(DINOv3FeatureExtractor):
    def forward_spatial(self, x, **kwargs) -> Any:
        outputs = self.image_encoder(x)
        last_hidden_states = outputs.last_hidden_state
        # cls_token = last_hidden_states[:, 0, :]
        patch_tokens = last_hidden_states[
            :, 1 + self.image_encoder.config.num_register_tokens :, :
        ]

        # Make spatial
        patch_tokens = patch_tokens.permute(0, 2, 1)
        patch_tokens = patch_tokens.reshape(
            patch_tokens.shape[0],
            patch_tokens.shape[1],
            int(patch_tokens.shape[2] ** 0.5),
            int(patch_tokens.shape[2] ** 0.5),
        )

        return patch_tokens

    def forward_spatial_to_pooled(self, x, **kwargs) -> Any:
        x = x.reshape(x.shape[0], x.shape[1], -1)
        return x.mean(dim=2)

    def get_normalize_values(self):
        return {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }

    def preprocess_tensor_image(self, img: torch.FloatTensor, **kwargs):
        x = F.resize(img, [224, 224], interpolation=F.InterpolationMode.BICUBIC)
        if isinstance(x, Image.Image):
            x = F.to_tensor(x)
        x = F.normalize(x, **self.get_normalize_values())
        return x

    def forward(self, x, **kwargs) -> Any:
        feats_spatial = self.forward_spatial(x, **kwargs)
        feats = self.forward_spatial_to_pooled(feats_spatial, **kwargs)
        return feats


class CUBDINOv3RecognitionModel(BaseRecognitionModelInterface):
    def __init__(self, class_num):
        feature_extractor = CUBDINOv3FeatureExtractor(
            hf_path="facebook/dinov3-vitb16-pretrain-lvd1689m",
            extraction_type=DINOFeatureExtractionType.PATCH,
        )
        classifier = BasicLinearClassifier(
            in_dim=feature_extractor.hidden_size, out_dim=class_num
        )
        super().__init__(feature_extractor=feature_extractor, classifier=classifier)
