from typing import Any

import torch
from transformers import CLIPImageProcessor, CLIPVisionModel

from .base import (
    BaseFeatureExtractorInterface,
    BaseRecognitionModelInterface,
    BasicLinearClassifier,
)


class CLIPFeatureExtractor(BaseFeatureExtractorInterface):
    def __init__(
        self,
        hf_path: str,
    ):
        super().__init__()
        self.hf_path = hf_path
        self.image_encoder = CLIPVisionModel.from_pretrained(hf_path)
        self.hidden_size = self.image_encoder.config.hidden_size
        self.image_encoder = self.image_encoder.vision_model

        self.image_processor = CLIPImageProcessor()

    def preprocess_image(self, img, **kwargs) -> Any:
        return self.image_processor(images=img, return_tensors="pt").pixel_values

    def forward(self, x, **kwargs):
        outputs = self.image_encoder(x)
        sequence_output = outputs.last_hidden_state
        feats = torch.mean(sequence_output[:, 1:, :], dim=1)
        return feats


class CLIPRecognitionModel(BaseRecognitionModelInterface):
    def __init__(self, class_num):
        feature_extractor = CLIPFeatureExtractor(
            hf_path="openai/clip-vit-base-patch32",
        )
        classifier = BasicLinearClassifier(
            in_dim=feature_extractor.hidden_size, out_dim=class_num
        )
        super().__init__(feature_extractor=feature_extractor, classifier=classifier)
