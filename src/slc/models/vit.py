from typing import Any

from transformers import AutoImageProcessor, ViTForImageClassification

from .base import (
    BaseFeatureExtractorInterface,
    BaseRecognitionModelInterface,
    BasicLinearClassifier,
)


class ViTFeatureExtractor(BaseFeatureExtractorInterface):
    def __init__(
        self,
        hf_path: str,
    ):
        super().__init__()
        self.hf_path = hf_path
        self.image_encoder = ViTForImageClassification.from_pretrained(hf_path)
        self.hidden_size = self.image_encoder.config.hidden_size
        self.image_encoder = self.image_encoder.vit
        self.image_processor = AutoImageProcessor.from_pretrained(hf_path)

    def preprocess_image(self, img, **kwargs) -> Any:
        return self.image_processor(img, return_tensors="pt")["pixel_values"]

    def forward(self, x, **kwargs) -> Any:
        outputs = self.image_encoder(x)
        sequence_output = outputs[0]
        feats = sequence_output[:, 0, :]
        return feats


class ViT_B_P16_224_RecognitionModel(BaseRecognitionModelInterface):
    def __init__(self, class_num):
        feature_extractor = ViTFeatureExtractor(
            hf_path="google/vit-base-patch16-224",
        )
        classifier = BasicLinearClassifier(
            in_dim=feature_extractor.hidden_size, out_dim=class_num
        )
        super().__init__(feature_extractor=feature_extractor, classifier=classifier)
