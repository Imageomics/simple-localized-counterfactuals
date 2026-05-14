from typing import Any

import torch
import torchvision.transforms.functional as F
from transformers import AutoImageProcessor, ResNetModel, ResNetForImageClassification
from torchvision.models import resnet50, ResNet50_Weights

from .base import (
    BaseFeatureExtractorInterface,
    BaseRecognitionModelInterface,
    BasicLinearClassifier,
)


class ResNetFeatureExtractor(BaseFeatureExtractorInterface):
    def __init__(
        self,
        hf_path: str,
    ):
        super().__init__()
        self.hf_path = hf_path
        self.image_encoder = ResNetModel.from_pretrained(hf_path)
        self.image_processor = AutoImageProcessor.from_pretrained(hf_path)
        self.hidden_size = self.image_encoder.config.hidden_sizes[-1]

    def preprocess_image(self, img, **kwargs) -> Any:
        return self.image_processor(img, return_tensors="pt")["pixel_values"]

    def get_normalize_values(self):
        return {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }

    def preprocess_tensor_image(self, img: torch.FloatTensor, **kwargs):
        img = F.resize(img, size=(224, 224), interpolation=F.InterpolationMode.BICUBIC)
        img = F.normalize(img, **self.get_normalize_values())
        return img

    def forward(self, x, **kwargs) -> Any:
        outputs = self.image_encoder(x, return_dict=True)
        pooled_output = outputs.pooler_output
        return pooled_output

    def forward_spatial(self, x, **kwargs) -> Any:
        outputs = self.image_encoder(x, return_dict=True)
        return outputs.last_hidden_state


class ResNet50RecognitionModel(BaseRecognitionModelInterface):
    def __init__(self, class_num, use_imagenet_classifier=False):
        hf_path = "microsoft/resnet-50"
        feature_extractor = ResNetFeatureExtractor(
            hf_path=hf_path,
        )
        if use_imagenet_classifier:
            resnet_model = ResNetForImageClassification.from_pretrained(hf_path)
            classifier = resnet_model.classifier
        else:
            classifier = BasicLinearClassifier(
                in_dim=feature_extractor.hidden_size, out_dim=class_num
            )

        super().__init__(feature_extractor=feature_extractor, classifier=classifier)


class ResNetPytorchFeatureExtractor(BaseFeatureExtractorInterface):
    def __init__(
        self,
        hf_path: str = None,
        weights: ResNet50_Weights = ResNet50_Weights.IMAGENET1K_V1,
    ):
        super().__init__()
        self.image_encoder = resnet50(weights=weights)
        self.avg_pool = self.image_encoder.avgpool

        self.image_processor = weights.transforms()

        self.hidden_size = 2048

    def preprocess_image(self, img, **kwargs) -> Any:
        x = self.image_processor(img)
        return x

    def get_normalize_values(self):
        return {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }

    def unnormalize(self, img: torch.FloatTensor) -> torch.FloatTensor:
        norm_vals = self.get_normalize_values()
        mean = norm_vals["mean"]
        std = norm_vals["std"]
        img = F.normalize(
            img, [-m / s for m, s in zip(mean, std)], [1 / s for s in std]
        )
        return img

    def preprocess_tensor_image(self, img: torch.FloatTensor, **kwargs):
        img = F.resize(img, size=(224, 224), interpolation=F.InterpolationMode.BICUBIC)
        img = F.normalize(img, **self.get_normalize_values())
        return img

    def forward(self, x, **kwargs) -> Any:
        h = self.forward_spatial(x)
        h = self.forward_spatial_to_pooled(h)
        return h

    def forward_spatial(self, x, **kwargs) -> Any:
        x = self.image_encoder.conv1(x)
        x = self.image_encoder.bn1(x)
        x = self.image_encoder.relu(x)
        x = self.image_encoder.maxpool(x)

        x = self.image_encoder.layer1(x)
        x = self.image_encoder.layer2(x)
        x = self.image_encoder.layer3(x)
        x = self.image_encoder.layer4(x)

        return x

    def forward_spatial_to_pooled(self, x, **kwargs):
        x = self.image_encoder.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class ResNet50PytorchRecognitionModel(BaseRecognitionModelInterface):
    def __init__(self, class_num, use_imagenet_classifier=False):
        feature_extractor = ResNetPytorchFeatureExtractor(
            hf_path=None, weights=ResNet50_Weights.IMAGENET1K_V1
        )

        if use_imagenet_classifier:
            classifier = feature_extractor.image_encoder.fc
        else:
            classifier = BasicLinearClassifier(
                in_dim=feature_extractor.hidden_size, out_dim=class_num
            )

        super().__init__(feature_extractor=feature_extractor, classifier=classifier)
