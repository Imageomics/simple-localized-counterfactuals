import torch
from saliency.core import BlurIG

from slc.models.base import BaseRecognitionModelInterface

from .base import BaseMaskGenerator


class BlurIntegratedGradientsMaskGenerator(BaseMaskGenerator):
    def __init__(self, classifier: BaseRecognitionModelInterface, **kwargs):
        super().__init__(classifier, **kwargs)
        self.saliency = BlurIG()

    def _call_fn(self, x_value_batch, call_model_args=None, expected_keys=None):
        x = torch.from_numpy(x_value_batch).permute((0, 3, 1, 2)).requires_grad_()
        target_class_idx = call_model_args["class_idx_str"]
        output = self.classifier(x.cuda())
        m = torch.nn.Softmax(dim=1)
        outputs = m(output)
        outputs = outputs[:, target_class_idx]
        grads = torch.autograd.grad(outputs, x, grad_outputs=torch.ones_like(outputs))
        grads = torch.movedim(grads[0], 1, 3)
        gradients = grads.detach().numpy()
        return {"INPUT_OUTPUT_GRADIENTS": gradients}

    def create_masks(
        self,
        x,
        target,
        threshold=0.8,
        only_positive=False,
        max_sigma=10,
        x_steps=100,
        do_threshold=True,
    ) -> torch.FloatTensor:
        x_in = x[0].permute((1, 2, 0)).cpu().numpy()
        saliency_map = self.saliency.GetMask(
            x_value=x_in,
            call_model_function=self._call_fn,
            call_model_args={"class_idx_str": target},
            max_sigma=max_sigma,
            steps=x_steps,
        )

        saliency_map = torch.from_numpy(saliency_map)
        saliency_map = saliency_map.mean(2)
        saliency_map = self._mask_from_saliency_map(
            saliency_map, threshold, only_positive, do_threshold
        )

        return saliency_map
