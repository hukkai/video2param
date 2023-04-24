from typing import Dict, List, Optional, Tuple
import torch
from detectron2.config import configurable
from detectron2.structures import ImageList, Instances, Boxes
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN


@META_ARCH_REGISTRY.register()
class GRiT(GeneralizedRCNN):
    @configurable
    def __init__(
        self,
        **kwargs):
        super().__init__(**kwargs)
        assert self.proposal_generator is not None

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        return ret

    @torch.no_grad()
    def inference(
        self,
        batched_inputs: Tuple[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        assert not self.training
        assert detected_instances is None
        num_images = len(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        proposals, _ = self.proposal_generator(images, features, None)
        results = []
        for idx in range(num_images):
            feature = dict(p3=features['p3'][idx:idx+1],
                           p4=features['p4'][idx:idx+1],
                           p5=features['p5'][idx:idx+1],
                           p6=features['p6'][idx:idx+1],
                           p7=features['p7'][idx:idx+1],)
            result, _ = self.roi_heads(feature, proposals[idx:idx+1])
            if do_postprocess:
                assert not torch.jit.is_scripting(), \
                    "Scripting is not supported for postprocess."
                result = GRiT._postprocess(result, 
                                         batched_inputs[idx:idx+1], 
                                         images.image_sizes[idx:idx+1])
            results += result
        return results

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        targets_task = batched_inputs[0]['task']
        for anno_per_image in batched_inputs:
            assert targets_task == anno_per_image['task']

        features = self.backbone(images.tensor)
        proposals, proposal_losses = self.proposal_generator(
            images, features, gt_instances)
        proposals, roihead_textdecoder_losses = self.roi_heads(
            features, proposals, gt_instances, targets_task=targets_task)

        losses = {}
        losses.update(roihead_textdecoder_losses)
        losses.update(proposal_losses)

        return losses