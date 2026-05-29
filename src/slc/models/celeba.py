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


class DiVEDenseNet121(torch.nn.Module):
    def __init__(self, path_to_weights):
        super().__init__()
        self.feat_extract = DenseNet121()
        self.classifier = torch.nn.Linear(self.feat_extract.output_size, 40)

        # load the model from the checkpoint
        state_dict = torch.load(path_to_weights, map_location="cpu")
        self.feat_extract.load_state_dict(state_dict["feat_extract"])
        self.classifier.load_state_dict(state_dict["classifier"])

    def forward(self, x):
        x = self.feat_extract(x)
        x = self.classifier(x)

        return x


class CelebAFeatureExtractor(BaseFeatureExtractorInterface):
    def __init__(self, base_model: DiVEDenseNet121):
        super().__init__()
        self.base_model: DiVEDenseNet121 = base_model

        self.hidden_size = base_model.feat_extract.output_size

    def preprocess_image(self, img, **kwargs) -> Any:
        x = F.resize(img, [128, 128], interpolation=F.InterpolationMode.BICUBIC)
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


class CelebARecognitionModel(BaseRecognitionModelInterface):
    def __init__(self, weight_path):
        model = DiVEDenseNet121(weight_path)
        feature_extractor = CelebAFeatureExtractor(model)
        classifier = model.classifier

        super().__init__(feature_extractor=feature_extractor, classifier=classifier)
