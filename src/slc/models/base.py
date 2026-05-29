from abc import abstractmethod
from typing import Any

import torch
import torch.nn as nn
from PIL import Image


class BaseFeatureExtractorInterface(nn.Module):
    @abstractmethod
    def forward(self, x, **kwargs) -> Any:
        raise NotImplementedError(
            f"forward function is not implemented in class {type(self).__name__}"
        )

    @abstractmethod
    def forward_spatial(self, x, **kwargs) -> Any:
        raise NotImplementedError(
            f"forward_spatial function is not implemented in class {type(self).__name__}"
        )

    @abstractmethod
    def forward_spatial_to_pooled(self, x, **kwargs) -> Any:
        raise NotImplementedError(
            f"forward_spatial_to_pooled function is not implemented in class {type(self).__name__}"
        )

    @abstractmethod
    def preprocess_image(self, img, **kwargs) -> Any:
        raise NotImplementedError(
            f"preprocess_image function is not implemented in class {type(self).__name__}"
        )

    @abstractmethod
    def preprocess_tensor_image(self, img: torch.FloatTensor, **kwargs) -> Any:
        raise NotImplementedError(
            f"preprocess_tensor_image function is not implemented in class {type(self).__name__}"
        )

    @abstractmethod
    def get_normalize_values(self) -> dict:
        raise NotImplementedError(
            f"get_normalize_values function is not implemented in class {type(self).__name__}"
        )


class BaseClassifierInterface(nn.Module):
    @abstractmethod
    def forward(self, x, **kwargs) -> Any:
        raise NotImplementedError(
            f"forward function is not implemented in class {type(self).__name__}"
        )


class BasicLinearClassifier(BaseFeatureExtractorInterface):
    def __init__(self, in_dim: int, out_dim: int, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x) -> torch.Tensor:  # type: ignore
        return self.linear(x)


class BaseRecognitionModelInterface(nn.Module):
    def __init__(self, feature_extractor, classifier):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def forward_features(self, x, **kwargs):
        h = self.feature_extractor(x, **kwargs)
        return h

    def forward_features_spatial(self, x, **kwargs):
        h = self.feature_extractor.forward_spatial(x, **kwargs)
        return h

    def forward_spatial_to_pooled(self, x, **kwargs):
        h = self.feature_extractor.forward_spatial_to_pooled(x, **kwargs)
        return h

    def forward_head(self, h, **kwargs):
        logits = self.classifier(h)
        return logits

    def forward(self, x, **kwargs) -> Any:
        h = self.forward_features(x, **kwargs)
        logits = self.forward_head(h, **kwargs)

        return logits

    def preprocess_image(self, img: Image.Image) -> Any:
        return self.feature_extractor.preprocess_image(img)

    def preprocess_tensor_image(self, img: torch.Tensor, **kwargs) -> Any:
        return self.feature_extractor.preprocess_tensor_image(img, **kwargs)
