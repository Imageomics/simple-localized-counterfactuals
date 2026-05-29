from enum import StrEnum
from typing import Any

import torch
from transformers import AutoImageProcessor, AutoModel

from .base import (
    BaseFeatureExtractorInterface,
    BaseRecognitionModelInterface,
    BasicLinearClassifier,
)


class DINOFeatureExtractionType(StrEnum):
    CLS = "cls"
    PATCH = "patch"
    CLS_AND_PATCH = "cls+patch"


class DINOFeatureExtractor(BaseFeatureExtractorInterface):
    def __init__(
        self,
        hf_path: str,
        extraction_type: DINOFeatureExtractionType = DINOFeatureExtractionType.CLS,
    ):
        super().__init__()
        self.hf_path = hf_path
        self.image_encoder = AutoModel.from_pretrained(hf_path)
        self.image_processor = AutoImageProcessor.from_pretrained(hf_path)
        self.extraction_type = extraction_type
        self.hidden_size = self.image_encoder.config.hidden_size
        if extraction_type == DINOFeatureExtractionType.CLS_AND_PATCH:
            self.hidden_size *= 2

    def preprocess_image(self, img, **kwargs) -> Any:
        return self.image_processor(img, return_tensors="pt")["pixel_values"][0]

    def _extract_last_features(self, cls_token, patch_tokens):
        match self.extraction_type:
            case DINOFeatureExtractionType.CLS:
                feats = cls_token
            case DINOFeatureExtractionType.PATCH:
                feats = patch_tokens.mean(dim=1)
            case DINOFeatureExtractionType.CLS_AND_PATCH:
                feats = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)
            case _:
                raise NotImplementedError(f"{self.extraction_type} is not implemented!")

        return feats


class DINOv2FeatureExtractor(DINOFeatureExtractor):
    def forward(self, x, **kwargs) -> Any:
        outputs = self.image_encoder(x)
        feats = outputs[0]
        cls_token = feats[:, 0, :]
        patch_tokens = feats[:, 1:, :]
        feats = self._extract_last_features(cls_token, patch_tokens)
        return feats


class DINOv3FeatureExtractor(DINOFeatureExtractor):
    def forward(self, x, **kwargs) -> Any:
        outputs = self.image_encoder(x)
        last_hidden_states = outputs.last_hidden_state
        cls_token = last_hidden_states[:, 0, :]
        patch_tokens = last_hidden_states[
            :, 1 + self.image_encoder.config.num_register_tokens :, :
        ]
        feats = self._extract_last_features(cls_token, patch_tokens)
        return feats


class DINOv2RecognitionModel(BaseRecognitionModelInterface):
    def __init__(self, class_num):
        feature_extractor = DINOv2FeatureExtractor(
            hf_path="facebook/dinov2-base",
            extraction_type=DINOFeatureExtractionType.CLS_AND_PATCH,
        )
        classifier = BasicLinearClassifier(
            in_dim=feature_extractor.hidden_size, out_dim=class_num
        )
        super().__init__(feature_extractor=feature_extractor, classifier=classifier)


class DINOv3RecognitionModel(BaseRecognitionModelInterface):
    def __init__(self, class_num):
        feature_extractor = DINOv3FeatureExtractor(
            hf_path="facebook/dinov3-vitb16-pretrain-lvd1689m",
            extraction_type=DINOFeatureExtractionType.CLS_AND_PATCH,
        )
        classifier = BasicLinearClassifier(
            in_dim=feature_extractor.hidden_size, out_dim=class_num
        )
        super().__init__(feature_extractor=feature_extractor, classifier=classifier)
