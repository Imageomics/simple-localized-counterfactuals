from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as F
from PIL import Image
from saev.nn import SparseAutoencoder
from tqdm import tqdm

from slc.models import (
    AutoencoderPipeline,
    BaseRecognitionModelInterface,
    CelebAHQRecognitionModel,
    CelebARecognitionModel,
    CUBDINOv3RecognitionModel,
)

from .base_edit import EditPipelineOutput
from .latent_edit import LatentEditPipeline, LatentEditPipelineConfigs


@dataclass
class SAELatentEditPipelineConfigs(LatentEditPipelineConfigs):
    sae_sparsity_lambda: float = 0.01


class SAELatentEditPipeline(LatentEditPipeline):
    def __init__(
        self,
        recognition_model: BaseRecognitionModelInterface,
        ae_model: AutoencoderPipeline,
        configs: Any,
        sae: SparseAutoencoder,
    ):
        super().__init__(recognition_model, ae_model, configs)
        self.sae: SparseAutoencoder = sae.cuda()
        self.sae.eval()

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
        recon_sae_feats: Any = None,
        sae_feature_mask: Any = None,
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
                recon_features = self.recognition_model.forward_features_spatial(
                    img_edit_pre.cuda()
                )
                recon_features_pooled = (
                    self.recognition_model.feature_extractor.forward_spatial_to_pooled(
                        recon_features * sae_feature_mask.unsqueeze(0)
                    )
                )
                output = self.recognition_model.forward_head(recon_features_pooled)

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

            if recon_sae_feats is not None:
                sae_edit_c_feats = self.sae(recon_features_pooled)[1]
                sae_sparsity_loss = l1_loss(sae_edit_c_feats, recon_sae_feats.detach())
                loss += sae_sparsity_loss * self.configs.sae_sparsity_lambda

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
            is_binary=is_binary,
            mask=mask,
            max_iters=25,
            peturb_strength=0,
            first_phase=True,
        )
        latent_mask = self._process_lig_mask(lig).cuda().detach()

        spatial_size = (7, 7)
        if isinstance(self.recognition_model, CelebARecognitionModel):
            spatial_size = (4, 4)
        elif isinstance(self.recognition_model, CelebAHQRecognitionModel):
            spatial_size = (8, 8)
        elif isinstance(self.recognition_model, CUBDINOv3RecognitionModel):
            spatial_size = (14, 14)

        sae_feature_mask = F.resize(
            latent_mask.mean(1), spatial_size, interpolation=F.InterpolationMode.BICUBIC
        )
        feat_size = self.recognition_model.feature_extractor.hidden_size
        sae_feature_mask = sae_feature_mask.repeat(feat_size, 1, 1)
        sae_feature_mask.shape

        sae_feature_mask -= sae_feature_mask.min()
        sae_feature_mask /= sae_feature_mask.max()

        with torch.no_grad():
            recon_features = self.recognition_model.forward_features_spatial(
                self.recognition_model.preprocess_tensor_image(xr).unsqueeze(0).cuda()
            )
            recon_features_pooled = (
                self.recognition_model.feature_extractor.forward_spatial_to_pooled(
                    recon_features * sae_feature_mask.unsqueeze(0)
                )
            )
            recon_sae_feats = self.sae(recon_features_pooled)[1]

        delta, _ = self._inner_forward(
            z,
            src_label=src_label,
            tgt_label=tgt_label,
            is_binary=is_binary,
            mask=latent_mask,
            max_iters=self.configs.max_iter_steps,
            peturb_strength=self.configs.peturb_strength,
            recon_sae_feats=recon_sae_feats,
            sae_feature_mask=sae_feature_mask,
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


@dataclass
class SAEFeatureLatentEditPipelineConfigs(LatentEditPipelineConfigs):
    sae_sparsity_lambda: float = 0.01


class SAEFeatureLatentEditPipeline(LatentEditPipeline):
    def __init__(
        self,
        recognition_model: BaseRecognitionModelInterface,
        ae_model: AutoencoderPipeline,
        configs: Any,
        sae: SparseAutoencoder,
    ):
        super().__init__(recognition_model, ae_model, configs)
        self.sae: SparseAutoencoder = sae.cuda()
        self.sae.eval()

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
        recon_sae_feats: Any = None,
        sae_feature_mask: Any = None,
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
                x_in = self.recognition_model.preprocess_tensor_image(img_edit_pret)
                recon_features = self.recognition_model.forward_features_spatial(
                    x_in.cuda()
                )
                recon_features_pooled = (
                    self.recognition_model.feature_extractor.forward_spatial_to_pooled(
                        recon_features
                    )
                )
                output = self.sae(recon_features_pooled)[1]
            else:
                recon_features = self.recognition_model.forward_features_spatial(
                    img_edit_pre.cuda()
                )
                recon_features_pooled = (
                    self.recognition_model.feature_extractor.forward_spatial_to_pooled(
                        recon_features * sae_feature_mask.unsqueeze(0)
                    )
                )
                output = self.sae(recon_features_pooled)[1]

            sm = torch.sigmoid(output)
            pred = torch.round(torch.sigmoid(output))
            pred = pred[:, tgt_label]

            if self._do_early_stop(pred, sm, src_label, tgt_label, is_binary):
                break

            cls_loss = self._calc_class_loss(
                output, sm, src_label, tgt_label, is_binary
            )

            sparsity_loss = l1_loss(delta.abs(), torch.zeros_like(delta).cuda())

            loss = cls_loss + sparsity_loss * self.configs.sparsity_lambda

            if recon_sae_feats is not None:
                sae_edit_c_feats = self.sae(recon_features_pooled)[1]
                sae_sparsity_loss = l1_loss(sae_edit_c_feats, recon_sae_feats.detach())
                loss += sae_sparsity_loss * self.configs.sae_sparsity_lambda

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
            is_binary=is_binary,
            mask=mask,
            max_iters=25,
            peturb_strength=0,
            first_phase=True,
        )
        latent_mask = self._process_lig_mask(lig).cuda().detach()

        spatial_size = (7, 7)
        if isinstance(self.recognition_model, CUBDINOv3RecognitionModel):
            spatial_size = (14, 14)

        sae_feature_mask = F.resize(
            latent_mask.mean(1), spatial_size, interpolation=F.InterpolationMode.BICUBIC
        )
        feat_size = self.recognition_model.feature_extractor.hidden_size
        sae_feature_mask = sae_feature_mask.repeat(feat_size, 1, 1)
        sae_feature_mask.shape

        sae_feature_mask -= sae_feature_mask.min()
        sae_feature_mask /= sae_feature_mask.max()

        with torch.no_grad():
            recon_features = self.recognition_model.forward_features_spatial(
                self.recognition_model.preprocess_tensor_image(xr).unsqueeze(0).cuda()
            )
            recon_features_pooled = (
                self.recognition_model.feature_extractor.forward_spatial_to_pooled(
                    recon_features * sae_feature_mask.unsqueeze(0)
                )
            )
            recon_sae_feats = self.sae(recon_features_pooled)[1]

        delta, _ = self._inner_forward(
            z,
            src_label=src_label,
            tgt_label=tgt_label,
            is_binary=is_binary,
            mask=latent_mask,
            max_iters=self.configs.max_iter_steps,
            peturb_strength=self.configs.peturb_strength,
            recon_sae_feats=recon_sae_feats,
            sae_feature_mask=sae_feature_mask,
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
