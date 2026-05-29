# Code adapted from: https://raw.githubusercontent.com/guillaumejs2403/ACE/refs/heads/main/models/dive/densenet.py

from typing import Any

import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
from torch.nn.functional import adaptive_avg_pool2d, relu

from slc.models.base import (
    BaseFeatureExtractorInterface,
    BaseRecognitionModelInterface,
)


class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class DenseNet121(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feat_extract = torchvision.models.densenet121(pretrained=False)
        self.feat_extract.classifier = Identity()
        self.output_size = 1024

    def forward(self, x):
        return self.feat_extract(x)


class DecisionDensenetModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feat_extract = DenseNet121()
        self.classifier = torch.nn.Linear(self.feat_extract.output_size, 3)

    def forward(self, x, before_sigmoid=True):
        x = self.feat_extract(x)
        x = self.classifier(x)


class CelebAHQFeatureExtractor(BaseFeatureExtractorInterface):
    def __init__(self, base_model: DecisionDensenetModel):
        super().__init__()
        self.base_model: DecisionDensenetModel = base_model

        self.hidden_size = base_model.feat_extract.output_size

    def preprocess_image(self, img, **kwargs) -> Any:
        x = F.resize(img, [256, 256], interpolation=F.InterpolationMode.BICUBIC)
        if isinstance(x, Image.Image):
            x = F.to_tensor(x)
        x = F.normalize(x, **self.get_normalize_values())
        return x

    def get_normalize_values(self):
        return {
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
        }

    def preprocess_tensor_image(self, img: torch.FloatTensor, **kwargs):
        return self.preprocess_image(img, **kwargs)

    def forward(self, x, **kwargs) -> Any:
        h = self.forward_spatial(x)
        h = self.forward_spatial_to_pooled(h)
        return h

    def forward_spatial(self, x, **kwargs) -> Any:
        features = self.base_model.feat_extract.feat_extract.features(x)
        out = relu(features, inplace=True)
        return out

    def forward_spatial_to_pooled(self, x, **kwargs):
        out = adaptive_avg_pool2d(x, (1, 1))
        out = torch.flatten(out, 1)
        return out


class CelebAHQRecognitionModel(BaseRecognitionModelInterface):
    def __init__(self, weight_path):
        model = DecisionDensenetModel()
        model.load_state_dict(
            torch.load(weight_path, map_location="cpu")["model_state_dict"]
        )
        feature_extractor = CelebAHQFeatureExtractor(model)
        classifier = model.classifier

        super().__init__(feature_extractor=feature_extractor, classifier=classifier)
