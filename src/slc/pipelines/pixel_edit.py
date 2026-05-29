from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as F
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

from slc.masks.ig import IntegratedGradientsMaskGenerator

from .base_edit import BaseEditPipelineInterface, EditPipelineOutput


@dataclass
class PixelEditPipelineConfigs:
    verbose: bool = False
    sparsity_lambda: float = 0.0
    lr: float = 0.1
    early_stop: bool = True
    early_stop_threshold: float = 0.95
    max_iter_steps: int = 500
    mask_threshold: float = 0.5
    use_gaussian_blur: bool = True
    gaussian_blur_sigma: float = 1
    aggregate_region_constraint: bool = True
    peturb_strength: float = 0.2
    class_loss_target: str = "target"
    only_positive_gradients: bool = False
    init_max_iter_steps: int = 25


class PixelEditPipeline(BaseEditPipelineInterface):
    def _inner_forward(
        self,
        x: torch.Tensor,
        src_label: int,
        tgt_label: int,
        mask: torch.Tensor,
        max_iters: int,
        peturb_strength: float,
        first_phase: bool = False,
        is_binary: bool = False,
    ) -> Any:
        delta = torch.zeros_like(x.detach(), requires_grad=True)
        optimizer = optim.Adam([delta], lr=self.configs.lr)
        softmax = nn.Softmax(dim=1)
        l1_loss = nn.L1Loss()

        tbar = tqdm(
            range(max_iters),
            desc="Creating Counterfactual",
            colour="#24a88c",
            disable=not self.configs.verbose,
        )
        integrated_gradients = torch.zeros_like(x).cpu()

        self._reset_min_successful_flips()
        for iter in tbar:
            optimizer.zero_grad()
            peturb = torch.normal(mean=0.0, std=1.0, size=delta.shape).cuda().detach()
            # peturb = torch.rand_like(delta).detach() * 2
            # peturb -= 1
            peturb *= peturb_strength
            x_edit = x.detach() + (delta + peturb) * mask.detach()
            x_edit = torch.clamp(x_edit, 0, 1)
            x_edit_input = self.recognition_model.preprocess_tensor_image(x_edit).cuda()
            output = self.recognition_model(x_edit_input.unsqueeze(0))

            if is_binary:
                sm = torch.sigmoid(output)
                pred = torch.round(torch.sigmoid(output))
                pred = pred[:, tgt_label]
            else:
                sm = softmax(output)
                val, pred = torch.max(output, 1)

            if self._do_early_stop(pred, sm, src_label, tgt_label, is_binary):
                break

            cls_loss = self._calc_class_loss(
                output, sm, src_label, tgt_label, is_binary
            )

            sparsity_loss = l1_loss(delta.abs(), torch.zeros_like(delta).cuda())

            loss = cls_loss + sparsity_loss * self.configs.sparsity_lambda
            loss.backward()
            optimizer.step()

            integrated_gradients += delta.grad.detach().cpu()

            tbar.set_postfix(
                {
                    "class_loss": cls_loss.item(),
                    "sparsity_loss": sparsity_loss.item(),
                    "predicted_class": pred.item(),
                    "src_conf": sm[0, src_label].item(),
                    "tgt_conf": sm[0, tgt_label].item(),
                }
            )

        x_edit = x + delta * mask.detach()
        x_edit = torch.clamp(x_edit, 0, 1)
        delta = x_edit - x

        return delta, integrated_gradients

    def _process_ig_mask(self, ig: torch.Tensor) -> torch.Tensor:
        if self.configs.only_positive_gradients:
            latent_mask = ig.detach()
            latent_mask[latent_mask < 0] = 0
        else:
            latent_mask = ig.abs()

        latent_mask /= latent_mask.max()
        latent_mask[latent_mask < self.configs.mask_threshold] = 0

        if self.configs.use_gaussian_blur:
            latent_mask = ndimage.gaussian_filter(
                latent_mask.numpy(), sigma=self.configs.gaussian_blur_sigma
            )
            latent_mask = torch.from_numpy(latent_mask)

        if self.configs.aggregate_region_constraint:
            latent_mask = latent_mask.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
        else:
            latent_mask = latent_mask.unsqueeze(0)
        return latent_mask

    def forward(
        self,
        x: Image.Image,
        src_label: int,
        tgt_label: int,
        is_binary: bool = False,
        **kwargs,
    ) -> EditPipelineOutput:
        with torch.no_grad():
            x = F.resize(x, size=(256, 265), interpolation=F.InterpolationMode.BICUBIC)
            x = F.to_tensor(x).cuda()

        mask_generator = IntegratedGradientsMaskGenerator(self.recognition_model)
        mask = mask_generator.create_masks(
            x.unsqueeze(0), target=src_label, do_threshold=False
        )
        latent_mask = self._process_ig_mask(mask)[0].cuda().detach()
        delta, _ = self._inner_forward(
            x,
            src_label=src_label,
            tgt_label=tgt_label,
            mask=latent_mask,
            max_iters=self.configs.max_iter_steps,
            peturb_strength=self.configs.peturb_strength,
            is_binary=is_binary,
        )

        with torch.no_grad():
            edit = torch.clamp(x + delta, 0, 1)
            edit_image = F.to_pil_image(edit)

        return EditPipelineOutput(
            full_mask=None,
            thresholded_mask=latent_mask,
            recon_image=None,
            edit_image=edit_image,
        )


class PixelNoMaskEditPipeline(PixelEditPipeline):
    def forward(
        self,
        x: Image.Image,
        src_label: int,
        tgt_label: int,
        is_binary: bool = False,
        **kwargs,
    ) -> EditPipelineOutput:
        with torch.no_grad():
            x = F.to_tensor(x).cuda()
            mask = torch.ones_like(x).cuda().detach()

        delta, _ = self._inner_forward(
            x,
            src_label=src_label,
            tgt_label=tgt_label,
            mask=mask,
            max_iters=self.configs.max_iter_steps,
            peturb_strength=self.configs.peturb_strength,
            is_binary=is_binary,
        )

        with torch.no_grad():
            edit = torch.clamp(x + delta, 0, 1)
            edit_image = F.to_pil_image(edit)

        return EditPipelineOutput(
            full_mask=None,
            thresholded_mask=mask,
            recon_image=None,
            edit_image=edit_image,
        )
