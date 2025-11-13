import torch
import torch.nn as nn
from typing import Dict

from .sam_no_prompt import SamAR

class DeepAR(nn.Module):
    def __init__(self, 
                 sam_model: SamAR,
                 mask_threshold: float = 0.0):
        super().__init__()
        self.sam_model = sam_model
        self.mask_threshold = mask_threshold

    def forward(self,
                x: torch.Tensor,
                multimask_output: bool = False,
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
        # Step 2: Predict masks using SAM
        outputs = {}
        batch_size = x.shape[0]
        batched_input = []
        for i in range(batch_size):
            batched_input.append({
                "image": x[i],
                "original_size": (x.shape[2], x.shape[3])
            })
        sam_outputs = self.sam_model(batched_input, multimask_output=multimask_output)

        #Extract masks
        masks = torch.stack([output['masks'].squeeze(1) for output in sam_outputs])
        outputs['masks'] = masks
        return outputs
    
    def get_binary_masks(self, x:torch.Tensor):
        """
        Get thresholded binary masks for inference/visualization.
        """
        with torch.no_grad():
            outputs = self.forward(x)
            binary_masks = (outputs['masks'] > self.mask_threshold).float()
        return binary_masks
