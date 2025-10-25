import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional

from .sam_no_prompt import SamAR
from .input_generator import IVT2RGB
from .map_reconstructor import Mask2ARMaps

class DeepAR(nn.Module):
    def __init__(self, 
                 sam_model: SamAR,
                 input_generator: IVT2RGB,
                 map_reconstructor: Mask2ARMaps):
        super().__init__()
        self.sam_model = sam_model
        self.input_generator = input_generator
        self.map_reconstructor = map_reconstructor

    def forward(self,
                x: torch.Tensor,
                multimask_output: bool = False,
                return_intermediate: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete DeepAR pipeline.

        Args:
            x: Input tensor (B, C, H, W)
            multimask_output: Whether to output multiple masks from SAM.
            return_intermediate: Whether to return intermediate outputs.

        Returns:
            A dictionary containing:
                - 'masks': Predicted masks from SAM.
                - 'ar_maps': Reconstructed AR maps.
                - 'intermediate': (Optional) Intermediate outputs if requested.
        """
        outputs = {}

        # Step 1: Generate input features
        x = self.input_generator(x)
        if return_intermediate:
            outputs['input_features'] = x

        # Step 2: Predict masks using SAM
        batch_size = x.shape[0]
        batched_input = []
        for i in range(batch_size):
            batched_input.append({
                "image": x[i],
                "original_size": (x.shape[2], x.shape[3])
            })
        sam_outputs = self.sam_model(batched_input, multimask_output=multimask_output)

        #Extract masks
        masks = torch.stack([output['masks'] for output in sam_outputs])
        if return_intermediate:
            outputs['masks'] = masks
            outputs['iou_predictions'] = torch.stack([output['iou_predictions'] for output in sam_outputs])

        # Step 3: AR Map Reconstruction
        reconstructed = self.maps_reconstructor(masks)
        outputs['output'] = reconstructed

        return outputs
