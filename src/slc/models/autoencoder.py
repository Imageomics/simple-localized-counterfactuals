from abc import abstractmethod
from typing import Any

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from diffusers import AutoencoderKL

from slc.constants import SDVAEPretrainedModels


def load_diffusers_kl_vae(
    pretrain_path="runwayml/stable-diffusion-v1-5",
    subfolder="vae",
    revision=None,
    variant=None,
) -> AutoencoderKL:
    vae = AutoencoderKL.from_pretrained(
        pretrain_path, subfolder=subfolder, revision=revision, variant=variant
    )
    return vae


class AutoencoderPipeline(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        r = self.decode(z)
        return r

    @abstractmethod
    def preprocess_image(self, img, **kwargs) -> Any:
        raise NotImplementedError(
            f"preprocess_image function is not implemented in class {type(self).__name__}"
        )

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            f"{type(self).__name__} needs to implement the {self.encode.__name__} method"
        )

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            f"{type(self).__name__} needs to implement the {self.decode.__name__} method"
        )


class SDVAEPipeline(AutoencoderPipeline):
    def __init__(
        self, pretrained_path: SDVAEPretrainedModels = SDVAEPretrainedModels.OSTRIS
    ):
        super().__init__()
        match pretrained_path:
            case SDVAEPretrainedModels.SD15:
                self.vae = (
                    load_diffusers_kl_vae(pretrain_path=pretrained_path.value)
                    .eval()
                    .cuda()
                )
            case SDVAEPretrainedModels.OSTRIS:
                self.vae = (
                    load_diffusers_kl_vae(
                        pretrain_path=pretrained_path.value, subfolder=""
                    )
                    .eval()
                    .cuda()
                )
            case _:
                raise NotImplementedError(
                    f"{pretrained_path} is not implemented. Please implement for {type(self).__name__}"
                )

    def _resize(self, x: torch.Tensor) -> torch.Tensor:
        output = F.resize(
            img=x, size=(512, 512), interpolation=T.InterpolationMode.BILINEAR
        )
        output = F.center_crop(output, output_size=(512, 512))
        return output

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        output = F.normalize(tensor=x, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return output

    def _unormalize(self, x: torch.Tensor) -> torch.Tensor:
        output = F.normalize(tensor=x, mean=[-1, -1, -1], std=[2, 2, 2])
        return output

    def preprocess_image(self, img, **kwargs) -> Any:
        xin = self._resize(img)
        xin = self._normalize(xin)
        return xin

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.vae.encode(x).latent_dist.mode()
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        out = self.vae.decode(z).sample
        out = torch.clamp(out, min=-1, max=1)
        out = self._unormalize(out)
        return out
