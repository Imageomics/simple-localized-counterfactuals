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
class LatentEditPipelineConfigs:
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


class LatentEditPipeline(BaseEditPipelineInterface):
    def _inner_forward(
        self,
        z: torch.Tensor,
        src_label: int,
        tgt_label: int,
        mask: torch.Tensor,
        max_iters: int,
        peturb_strength: float,
        first_phase: bool = False,
        is_binary: bool = False,
    ) -> Any:
        delta = torch.zeros_like(z, requires_grad=True).cuda()
        optimizer = optim.Adam([delta], lr=self.configs.lr)
        softmax = nn.Softmax(dim=1)
        l1_loss = nn.L1Loss()

        tbar = tqdm(
            range(max_iters),
            desc="Creating Counterfactual",
            colour="#24a88c",
            disable=not self.configs.verbose,
        )
        latent_integrated_gradients = torch.zeros_like(z).cpu()

        self._reset_min_successful_flips()
        for iter in tbar:
            optimizer.zero_grad()
            peturb = torch.normal(mean=0.0, std=1.0, size=delta.shape).cuda().detach()
            # peturb = torch.rand_like(delta).detach() * 2
            # peturb -= 1
            peturb *= peturb_strength
            z_edit = z + (delta + peturb) * mask
            img_edit: torch.Tensor = self.ae_model.decode(z_edit)
            img_edit_pre = self.recognition_model.preprocess_tensor_image(img_edit)
            if first_phase:
                img_edit_pret = self.ae_model(self.ae_model.preprocess_image(img_edit))
                output = self.recognition_model(
                    self.recognition_model.preprocess_tensor_image(img_edit_pret)
                )
            else:
                output = self.recognition_model(img_edit_pre)

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

            latent_integrated_gradients += delta.grad.detach().cpu()

            tbar.set_postfix(
                {
                    "class_loss": cls_loss.item(),
                    "sparsity_loss": sparsity_loss.item(),
                    "predicted_class": pred.item(),
                    "src_conf": sm[0, src_label].item(),
                    "tgt_conf": sm[0, tgt_label].item(),
                }
            )

        return delta * mask.detach(), latent_integrated_gradients

    def _process_lig_mask(self, lig: torch.Tensor) -> torch.Tensor:
        if self.configs.only_positive_gradients:
            latent_mask = lig[0].detach()
            latent_mask[latent_mask < 0] = 0
        else:
            latent_mask = lig[0].abs()

        if self.configs.aggregate_region_constraint:
            latent_mask = latent_mask.sum(0)

        if self.configs.use_gaussian_blur:
            latent_mask = ndimage.gaussian_filter(
                latent_mask.numpy(), sigma=self.configs.gaussian_blur_sigma
            )
            latent_mask = torch.from_numpy(latent_mask)

        latent_mask /= latent_mask.max()
        latent_mask[latent_mask < self.configs.mask_threshold] = 0

        if self.configs.aggregate_region_constraint:
            latent_mask = latent_mask.unsqueeze(0).unsqueeze(0).repeat(1, 16, 1, 1)
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
            x = self.ae_model.preprocess_image(F.to_tensor(x)).unsqueeze(0).cuda()
            z = self.ae_model.encode(x).detach()
            xr = self.ae_model.decode(z)[0]
            recon = F.to_pil_image(xr)

            mask = torch.ones_like(z).cuda().detach()

        delta, lig = self._inner_forward(
            z,
            src_label=src_label,
            tgt_label=tgt_label,
            mask=mask,
            max_iters=self.configs.init_max_iter_steps,
            peturb_strength=0,
            first_phase=True,
            is_binary=is_binary,
        )
        latent_mask = self._process_lig_mask(lig).cuda().detach()
        delta, _ = self._inner_forward(
            z,
            src_label=src_label,
            tgt_label=tgt_label,
            mask=latent_mask,
            max_iters=self.configs.max_iter_steps,
            peturb_strength=self.configs.peturb_strength,
            is_binary=is_binary,
        )

        with torch.no_grad():
            edit = self.ae_model.decode(z + delta)[0]
            edit_image = F.to_pil_image(edit)

        return EditPipelineOutput(
            full_mask=lig,
            thresholded_mask=latent_mask,
            recon_image=recon,
            edit_image=edit_image,
        )


class LatentNoMaskEditPipeline(LatentEditPipeline):
    def forward(
        self,
        x: Image.Image,
        src_label: int,
        tgt_label: int,
        is_binary: bool = False,
        **kwargs,
    ) -> EditPipelineOutput:
        with torch.no_grad():
            x = self.ae_model.preprocess_image(F.to_tensor(x)).unsqueeze(0).cuda()
            z = self.ae_model.encode(x).detach()
            xr = self.ae_model.decode(z)[0]
            recon = F.to_pil_image(xr)

            mask = torch.ones_like(z).cuda().detach()

        delta, _ = self._inner_forward(
            z,
            src_label=src_label,
            tgt_label=tgt_label,
            mask=mask,
            max_iters=self.configs.max_iter_steps,
            peturb_strength=self.configs.peturb_strength,
            is_binary=is_binary,
        )

        with torch.no_grad():
            edit = self.ae_model.decode(z + delta)[0]
            edit_image = F.to_pil_image(edit)

        return EditPipelineOutput(
            full_mask=None,
            thresholded_mask=None,
            recon_image=recon,
            edit_image=edit_image,
        )


class LatentIGMaskEditPipeline(LatentEditPipeline):
    def _process_ig_mask(self, ig: torch.Tensor):
        ig = F.resize(ig, (64, 64), interpolation=F.InterpolationMode.BICUBIC)

        if self.configs.only_positive_gradients:
            latent_mask = ig.detach()
            latent_mask[latent_mask < 0] = 0
        else:
            latent_mask = ig.abs()

        if self.configs.use_gaussian_blur:
            latent_mask = ndimage.gaussian_filter(
                latent_mask.numpy(), sigma=self.configs.gaussian_blur_sigma
            )
            latent_mask = torch.from_numpy(latent_mask)

        latent_mask /= latent_mask.max()
        latent_mask[latent_mask < self.configs.mask_threshold] = 0

        latent_mask = latent_mask.unsqueeze(0).unsqueeze(0).repeat(1, 16, 1, 1)

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
            x = self.ae_model.preprocess_image(F.to_tensor(x)).unsqueeze(0).cuda()
            z = self.ae_model.encode(x).detach()
            xr = self.ae_model.decode(z)[0]
            recon = F.to_pil_image(xr)

            mask = torch.ones_like(z).cuda().detach()

        class WrapperModel(nn.Module):
            def __init__(self, ae, classifier):
                super().__init__()
                self.ae = ae.cpu()
                self.rm = classifier.cpu()

            def forward(self, z):
                h = self.ae.decode(z)
                im = self.rm.preprocess_tensor_image(h)
                return self.rm(im)

        # Need to make CPU here due to high VRAM usage of IG
        # TODO find a way to optimize this
        wrapper_model = WrapperModel(self.ae_model, self.recognition_model).cpu()
        mask_generator = IntegratedGradientsMaskGenerator(wrapper_model)
        mask = mask_generator.create_masks(
            z.cpu(), target=src_label, do_threshold=False
        )

        self.ae_model = self.ae_model.cuda()
        self.recognition_model = self.recognition_model.cuda()

        latent_mask = self._process_ig_mask(mask).cuda().detach()

        delta, _ = self._inner_forward(
            z,
            src_label=src_label,
            tgt_label=tgt_label,
            mask=latent_mask,
            max_iters=self.configs.max_iter_steps,
            peturb_strength=self.configs.peturb_strength,
            is_binary=is_binary,
        )

        with torch.no_grad():
            edit = self.ae_model.decode(z + delta)[0]
            edit_image = F.to_pil_image(edit)

        return EditPipelineOutput(
            full_mask=None,
            thresholded_mask=latent_mask,
            recon_image=recon,
            edit_image=edit_image,
        )


class LatentInterpolateLIGEditPipeline(LatentEditPipeline):
    def create_lig_mask(self, z, z_edit, lbl, steps=50):
        alpha = 1 / steps
        mask = torch.zeros_like(z).cpu()
        direction = z_edit - z
        for i in tqdm(
            range(steps + 1),
            desc="Performing Latent Integrated Gradients",
            disable=not self.configs.verbose,
        ):
            self.recognition_model.zero_grad()
            self.ae_model.zero_grad()
            delta = alpha * i
            input_latent = (z + direction * delta).clone().detach().requires_grad_()
            created_image = self.ae_model.decode(input_latent)
            input_image = self.recognition_model.preprocess_tensor_image(created_image)
            logits = self.recognition_model(input_image)

            (-logits[:, lbl]).backward()

            mask += input_latent.grad.detach().cpu()

        return mask

    def forward(
        self,
        x: Image.Image,
        src_label: int,
        tgt_label: int,
        is_binary: bool = False,
        **kwargs,
    ) -> EditPipelineOutput:
        with torch.no_grad():
            x = self.ae_model.preprocess_image(F.to_tensor(x)).unsqueeze(0).cuda()
            z = self.ae_model.encode(x).detach()
            xr = self.ae_model.decode(z)[0]
            recon = F.to_pil_image(xr)

            mask = torch.ones_like(z).cuda().detach()

        delta, _ = self._inner_forward(
            z,
            src_label=src_label,
            tgt_label=tgt_label,
            mask=mask,
            max_iters=self.configs.init_max_iter_steps,
            peturb_strength=0,
            first_phase=True,
            is_binary=is_binary,
        )

        lig = self.create_lig_mask(z, z + delta, tgt_label, steps=50)

        latent_mask = self._process_lig_mask(lig).cuda().detach()
        delta, _ = self._inner_forward(
            z,
            src_label=src_label,
            tgt_label=tgt_label,
            mask=latent_mask,
            max_iters=self.configs.max_iter_steps,
            peturb_strength=self.configs.peturb_strength,
            is_binary=is_binary,
        )

        with torch.no_grad():
            edit = self.ae_model.decode(z + delta)[0]
            edit_image = F.to_pil_image(edit)

        return EditPipelineOutput(
            full_mask=lig,
            thresholded_mask=latent_mask,
            recon_image=recon,
            edit_image=edit_image,
        )
