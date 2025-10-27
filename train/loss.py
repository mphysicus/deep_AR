import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: raw model outputs (B,H,W) or (B, 1, H, W)
        targets: ground truth binary masks (B,H,W) or (B, 1, H, W)
        """
        if logits.ndim == 3:
            logits = logits.unsqueeze(1)
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)

        probs = torch.sigmoid(logits)

        probs = probs.contiguous().view(logits.shape[0], -1)
        targets = targets.contiguous().view(targets.shape[0], -1).float()

        intersection = (probs * targets).sum(dim=(1))
        denominator = (probs.pow(2).sum(dim=1) + targets.pow(2).sum(dim=1))

        dice_score = (2 * intersection + self.smooth) / (denominator + self.smooth)
        return 1 - dice_score.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        logits: raw model outputs (B,H,W) or (B, 1, H, W)
        targets: ground truth binary masks (B,H,W) or (B, 1, H, W)
        """
        #Flatten the tensors
        logits = logits.view(-1)
        targets = targets.view(-1)

        probs = torch.sigmoid(logits)

        pt = torch.where(targets == 1, probs, 1 - probs)

        alpha_t = torch.where(targets==1, self.alpha, 1 - self.alpha)

        # Compute focal loss
        focal_term = (1 - pt) ** self.gamma
        loss = -alpha_t * focal_term * torch.log(pt + 1e-8)

        return loss.mean()
    
class CombinedLoss(nn.Module):
    def __init__(self, alpha=20, focal_alpha=0.25, focal_gamma=2.0):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.alpha = alpha

    def forward(self, logits, targets):
        dice = self.dice_loss(logits, targets)
        focal = self.focal_loss(logits, targets)
        return (self.alpha * focal) + dice