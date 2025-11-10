import torch

def get_scheduler(optimizer, args):
    """
    Creates and returns a LambdaLR scheduler with warmup and cosine annealing
    based on the provided arguments.
    """
    cosine_epochs = max(1, args.epochs - args.warmup_epochs)

    if args.scratch_lr == 0:
         raise ValueError("Scratch learning rate cannot be zero for scheduler.")
    
    start_lr_factor = args.warmup_start_lr / args.scratch_lr
    def lr_lambda_pretrained(epoch):
            """
            LR scheduler for pre-trained parameters.
            Constant during warmup, cosine annealing afterwards.
            """
            if epoch < args.warmup_epochs:
                return 1.0
            else:
                progress = (epoch - args.warmup_epochs) / cosine_epochs
                return 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * progress))).item()
            
    def lr_lambda_scratch(epoch):
        """
        LR scheduler for parameters trained from scratch.
        Linear warmup, then cosine annealing.
        """
        if epoch < args.warmup_epochs:
            # Linear warmup
            if args.warmup_epochs == 0: return 1.0 #Avoid division by zero
            return start_lr_factor + (1.0 - start_lr_factor) * (epoch / args.warmup_epochs)
        else:
            progress = (epoch - args.warmup_epochs) / cosine_epochs
            return 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * progress))).item()

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lr_lambda=[lr_lambda_pretrained, lr_lambda_scratch])
    return scheduler