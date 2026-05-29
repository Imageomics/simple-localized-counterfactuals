import torch
from tqdm import tqdm

from slc.external.AGI.AGI_main import pgd_step
from slc.models.base import BaseRecognitionModelInterface

from .base import BaseMaskGenerator


class AdversarialGradientIntegrationMaskGenerator(BaseMaskGenerator):
    def __init__(self, classifier: BaseRecognitionModelInterface, **kwargs):
        super().__init__(classifier, **kwargs)

    def create_masks(
        self,
        x,
        target,
        threshold=0.8,
        only_positive=False,
        epsilon=0.05,
        top_k=10,
        x_steps=100,
        do_threshold=True,
    ) -> torch.FloatTensor:
        target = torch.tensor([target]).cuda()

        with torch.no_grad():
            output = self.classifier(x.cuda())
            top_ids = torch.argsort(output, dim=1)[0]

        top_k = min(top_k, output.shape[1])
        top_ids = top_ids[:top_k]
        step_grad = 0
        for l in tqdm(top_ids, "Peforming Adversarial Gradient Integration"):
            targeted = torch.tensor([l]).cuda()
            if targeted.item() == target.item():
                continue  # we don't want to attack to the predicted class.

            delta, perturbed_image = pgd_step(
                x, epsilon, self.classifier, target, targeted, max_iter=x_steps
            )
            step_grad += delta

        adv_ex = step_grad.squeeze().detach().cpu().numpy()  # / topk

        saliency_map = torch.from_numpy(adv_ex)
        saliency_map = saliency_map.mean(0)
        saliency_map = self._mask_from_saliency_map(
            saliency_map, threshold, only_positive, do_threshold=do_threshold
        )

        return saliency_map
