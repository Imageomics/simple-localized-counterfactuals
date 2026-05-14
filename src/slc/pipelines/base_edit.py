from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

from PIL import Image

from slc.models import AutoencoderPipeline, BaseRecognitionModelInterface


@dataclass
class EditPipelineOutput:
    full_mask: Any
    thresholded_mask: Any
    recon_image: Image.Image
    edit_image: Image.Image


class BaseEditPipelineInterface:
    def __init__(
        self,
        recognition_model: BaseRecognitionModelInterface,
        ae_model: AutoencoderPipeline,
        configs: Any,
    ):
        self.recognition_model = recognition_model.cuda()
        self.recognition_model.eval()
        self.ae_model = ae_model.cuda()
        self.ae_model.eval()
        self.configs = configs

    def __call__(self, x, **kwargs) -> EditPipelineOutput:
        return self.forward(x, **kwargs)

    @abstractmethod
    def forward(
        self, x: Image.Image, src_label: int, tgt_label: int, **kwargs
    ) -> EditPipelineOutput:
        raise NotImplementedError(
            f"forward function is not implemented in class {type(self).__name__}"
        )

    def _calc_class_loss(self, output, sm, src_label, tgt_label, is_binary=False):
        if is_binary:
            if src_label == 0:
                cls_loss = -output[0, tgt_label]
            elif src_label == 1:
                cls_loss = output[0, tgt_label]
            else:
                raise ValueError(
                    f"For binary tasks, the src_label should be either -1 or 1 for the gt label of the image we are editing. Value: {src_label}"
                )

            return cls_loss

        match self.configs.class_loss_target:
            case "target":
                cls_loss = -output[0, tgt_label]
            case "sm_target":
                cls_loss = -sm[0, tgt_label]
            case "source":
                cls_loss = output[0, src_label]
            case "both":
                cls_loss = output[0, src_label] - output[0, tgt_label]
            case _:
                raise NotImplementedError(
                    f"{self.configs.class_loss_target} is not implemented!"
                )
        return cls_loss

    def _reset_min_successful_flips(self):
        self.min_successful_flips = 5

    def _do_early_stop(self, pred, sm, src_label, tgt_label, is_binary=False):
        if not self.configs.early_stop:
            return False

        if is_binary:
            flipped_prediction = False
            greater_than_threshold = False
            flipped_prediction = pred.item() == (1 - src_label)
            if src_label == 0:
                greater_than_threshold = (
                    sm[0, tgt_label] > self.configs.early_stop_threshold
                )
            elif src_label == 1:
                greater_than_threshold = sm[0, tgt_label] < (
                    1 - self.configs.early_stop_threshold
                )
            else:
                raise ValueError(
                    f"For binary tasks, the src_label should be either 0 or 1 for the gt label of the image we are editing. Value: {src_label}"
                )

            if flipped_prediction and greater_than_threshold:
                self.min_successful_flips -= 1
            else:
                self._reset_min_successful_flips()

        else:
            if (
                self.configs.class_loss_target in ["sm_target", "target", "both"]
                and pred.item() == tgt_label
                and sm[0, tgt_label].item() > self.configs.early_stop_threshold
            ):
                self.min_successful_flips -= 1
            elif (
                self.configs.class_loss_target == "source" and pred.item() != src_label
            ):
                self.min_successful_flips -= 1
            else:
                self._reset_min_successful_flips()

        if self.min_successful_flips <= 0:
            return True

        return False
