import torch

def get_scheduler(optimizer, args, total_steps):
    """
    Creates and returns a LambdaLR scheduler with warmup and cosine annealing
    based on the provided arguments. Scheduler is called per training step.
    
    Args:
        optimizer: PyTorch optimizer
        args: Training arguments
        total_steps: Total number of training steps
    """
    warmup_steps = args.warmup_steps
    cosine_steps = max(1, total_steps - warmup_steps)

    if args.scratch_lr == 0:
         raise ValueError("Scratch learning rate cannot be zero for scheduler.")
    
    start_lr_factor = args.warmup_start_lr / args.scratch_lr
    min_lr_factor_scratch = args.min_lr / args.scratch_lr
    min_lr_factor_pretrained = args.min_lr / args.pretrained_lr if args.pretrained_lr != 0 else 0.0
    
    def lr_lambda_pretrained(step):
            """
            LR scheduler for pre-trained parameters.
            Constant during warmup, cosine annealing afterwards.
            """
            if step < warmup_steps:
                return 1.0
            else:
                progress = (step - warmup_steps) / cosine_steps
                cosine_decay = 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * progress))).item()
                return min_lr_factor_pretrained + (1.0 - min_lr_factor_pretrained) * cosine_decay
            
    def lr_lambda_scratch(step):
        """
        LR scheduler for parameters trained from scratch.
        Linear warmup, then cosine annealing.
        """
        if step < warmup_steps:
            if warmup_steps == 0: 
                return 1.0  # Avoid division by zero
            return start_lr_factor + (1.0 - start_lr_factor) * (step / warmup_steps)
        else:
            progress = (step - warmup_steps) / cosine_steps
            cosine_decay = 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * progress))).item()
            return min_lr_factor_scratch + (1.0 - min_lr_factor_scratch) * cosine_decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lr_lambda=[lr_lambda_pretrained, lr_lambda_scratch])
    return scheduler