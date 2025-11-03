from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.amp import GradScaler, autocast
import argparse
import os
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deep_ar.build_deep_ar import deep_ar_model_registry
from deep_ar.data.datasets import ARTrainingDataset
from loss import CombinedLoss

# Import LoRA utilities
from lora.utils import apply_lora_to_sam

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

def train_one_epoch(model, dataloader, optimizer, criterion, rank, scaler, epoch, use_wandb):
    model.train()
    total_train_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        images = batch['image'].to(rank, non_blocking=True)
        masks = batch['gt_mask'].to(rank, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda"):
            outputs = model(images)
            logits = outputs['output']
            loss = criterion(logits, masks)
        
        total_train_loss += loss.item()

        scaler.scale(loss).backward()

        del outputs, logits
        scaler.step(optimizer)
        scaler.update()


        if rank==0 and batch_idx % 10 == 0:
            print(f"Rank {rank}, Batch {batch_idx}, Loss: {loss.item()}")
            
            # Log batch-level metrics to wandb if enabled
            if use_wandb:
                wandb.log({
                    "batch_loss": loss.item(),
                    "batch": epoch * len(dataloader) + batch_idx
                })

    return total_train_loss / len(dataloader)

def validate(model, dataloader, criterion, rank):
    """
    Validate the model on the validation dataset.
    """
    model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(rank, non_blocking=True)
            masks = batch['gt_mask'].to(rank, non_blocking=True)

            with autocast("cuda"):
                outputs = model(images)
                logits = outputs['output']
                loss = criterion(logits, masks)

            total_val_loss += loss.item()

            del outputs, logits, loss

    avg_val_loss = total_val_loss / len(dataloader)

    #Synchronize the validation loss across all processes
    avg_val_loss_tensor = torch.tensor(avg_val_loss, device=rank)
    dist.all_reduce(avg_val_loss_tensor, op=dist.ReduceOp.AVG)

    return avg_val_loss_tensor.item()

def train_ddp(rank, world_size, args):
    setup_ddp(rank, world_size)

    # Initialize wandb only on rank 0 and if enabled
    use_wandb = (args.use_wandb.lower() == 'true') and WANDB_AVAILABLE
    if rank == 0 and use_wandb:
        config_dict = {
            "model_type": args.model_type,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "pretrained_lr":args.pretrained_lr,
            "scratch_lr": args.scratch_lr,
            "warmup_epochs": args.warmup_epochs,
            "warmup_start_lr": args.warmup_start_lr,
            "world_size": world_size,
            "optimizer": "AdamW",
            "scheduler": "Warmup + CosineAnnealingLR",
            "peft_method": args.peft_method,
            "use_gradient_checkpointing": args.use_gradient_checkpointing,
        }
        if args.peft_method in ['lora']:
            config_dict.update({
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
            })
        
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=config_dict,
            tags=args.wandb_tags.split(',') if args.wandb_tags else None,
        )
        print("✓ W&B tracking enabled")
    elif rank == 0:
        if not WANDB_AVAILABLE and args.use_wandb:
            print("⚠ Warning: wandb not installed, tracking disabled")
        else:
            print("✓ W&B tracking disabled (user choice)")

    #Build model
    model_builder = deep_ar_model_registry[args.model_type]
    model = model_builder(checkpoint=None, use_gradient_checkpointing=args.use_gradient_checkpointing)

    if args.original_sam_checkpoint is not None:
        if rank == 0:
            print(f"Loading SAM's image encoder and mask decoder checkpoint from {args.original_sam_checkpoint}...")
        
        state_dict = torch.load(args.original_sam_checkpoint, map_location=f'cuda:{rank}')
        adapted_state_dict = {"sam_model." + k: v for k, v in state_dict.items()}
        
        load_result = model.load_state_dict(adapted_state_dict, strict=False)

        if rank == 0:
            print(f"Successfully loaded pretrained SAM weights into DeepAR model.")
            if load_result.missing_keys:
                print(f"Ignored {len(load_result.missing_keys)} missing keys.")
            if load_result.unexpected_keys:
                print(f"Warning: Found {len(load_result.unexpected_keys)} unexpected keys in checkpoint.")


    # Apply PEFT method if requested
    if args.peft_method == 'lora':
        if rank == 0:
            print(f"✓ Applying LoRA (rank={args.lora_rank}, alpha={args.lora_alpha}, dropout={args.lora_dropout})")
        
        for param in model.sam_model.parameters():
            param.requires_grad = False

        LORA_TARGET_MODULES = [
            "attn.qkv",
            "attn.proj",
            "mlp.lin1",  # Present in both image encoder and mask decoder
            "mlp.lin2",  # Present in both image encoder and mask decoder

            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",

            "output_hypernetworks_mlps",
            "iou_prediction_head",
        ]
        # Apply LoRA to the SAM model inside DeepAR
        apply_lora_to_sam(
            model.sam_model,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            verbose=(rank == 0),
            target_layers= LORA_TARGET_MODULES
        )
        
        if hasattr(model.sam_model, 'no_mask_embedding'):
            model.sam_model.no_mask_embedding.requires_grad = True

        if hasattr(model.sam_model, 'positional_encoding'):
            model.sam_model.positional_encoding.requires_grad = True

        if hasattr(model.sam_model, 'image_encoder') and hasattr(model.sam_model.image_encoder, 'patch_embed'):
            for param in model.sam_model.image_encoder.patch_embed.parameters():
                param.requires_grad = True
            if rank == 0:
                print("✓ Made patch embedding parameters (in ViT) trainable.")
        
        if rank == 0:
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"✓ Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    
    elif args.peft_method == 'none':
        if rank == 0:
            print("✓ Using full fine-tuning (all parameters trainable)")
    else:
        raise ValueError(f"Unknown PEFT method: {args.peft_method}. Choose from: none, lora")

    model = model.to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    params_scratch = []
    params_pretrained = []

    scratch_param_keywords = [
        'patch_embed',
        'positional_encoding',
        'no_mask_embedding',
        'input_generator',
        'map_reconstructor'
    ]
    if  rank == 0:
        print(f"Building parameter groups for mode: {args.peft_method}...")

    for name, param in model.module.named_parameters():
        if not param.requires_grad:
            continue

        is_scratch = any(keyword in name for keyword in scratch_param_keywords)

        if is_scratch:
            params_scratch.append(param)
        else:
            params_pretrained.append(param)
    
    if rank == 0:
        print(f"✓ Found {len(params_pretrained)} 'pre-trained' parameter tensors.")
        print(f"✓ Found {len(params_scratch)} 'from scratch' parameter tensors.")
        print(f"  Pre-trained LR (LoRA or SAM blocks): {args.pretrained_lr}")
        print(f"  Scratch LR (CNNs, Embeds): {args.scratch_lr} (warming up from {args.warmup_start_lr} for {args.warmup_epochs} epochs)")

    param_groups = [
        {'params': params_pretrained, 'lr': args.pretrained_lr, 'name': 'pretrained'},
        {'params': params_scratch, 'lr': args.scratch_lr, 'name': 'scratch'},
    ]

    train_input_files = sorted(list(Path(args.train_input_dir).glob("*.nc")))
    val_input_files = sorted(list(Path(args.val_input_dir).glob("*.nc")))

    train_gt_files = sorted(list(Path(args.train_gt_dir).glob("*.nc")))
    val_gt_files = sorted(list(Path(args.val_gt_dir).glob("*.nc")))
    
    train_dataset = ARTrainingDataset(input_files=train_input_files, gt_files=train_gt_files)
    val_dataset = ARTrainingDataset(input_files=val_input_files, gt_files=val_gt_files)

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(param_groups)

    # LR scheduler with warmup
    cosine_epochs = max(1, args.epochs - args.warmup_epochs)
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
            #Linear warmup
            if args.warmup_epochs == 0: return 1.0 #Avoid division by zero
            return start_lr_factor + (1.0 - start_lr_factor) * (epoch / args.warmup_epochs)
        else:
            progress = (epoch - args.warmup_epochs) / cosine_epochs
            return 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * progress))).item()

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lr_lambda=[lr_lambda_pretrained, lr_lambda_scratch])
    
    scaler = GradScaler()

    # Watch model (only on rank 0 and if wandb enabled)
    if rank == 0 and use_wandb:
        wandb.watch(model, log="all", log_freq=100)
    
    if rank == 0 and args.checkpoint:
        os.makedirs(args.checkpoint, exist_ok=True)

    best_val_loss = torch.inf
    no_improve_epochs = 0

    stop_training_signal = torch.tensor(0, device=rank)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)

        current_lr_pretrained = optimizer.param_groups[0]['lr']
        current_lr_scratch = optimizer.param_groups[1]['lr']

        epoch_loss = train_one_epoch(model, train_loader, optimizer, criterion, rank, scaler, epoch, use_wandb)

        if rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Training Loss: {epoch_loss}")
            print(f"  Learning Rates - Pretrained: {current_lr_pretrained}, Scratch: {current_lr_scratch}")

        val_loss = validate(model, val_loader, criterion, rank)

        if rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Validation Loss: {val_loss}")
            
            # Log epoch-level metrics to wandb if enabled
            if use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": epoch_loss,
                    "val_loss": val_loss,
                    "lr_pretrained": current_lr_pretrained,
                    "lr_scratch": current_lr_scratch
                })
           
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_epochs = 0
                checkpoint_path = f"{args.checkpoint}/best_model_epoch_{epoch+1}_rank{args.lora_rank}_batch{args.batch_size}.pth"
                
                # Save appropriate state dict based on training mode
                if args.peft_method == 'lora':
                    # Save full model state dict (including LoRA weights)
                    torch.save(model.module.state_dict(), checkpoint_path)
                    print(f"Saved Best Model with Validation Loss: {best_val_loss}")
                else:
                    # Save full model state dict
                    torch.save(model.module.state_dict(), checkpoint_path)
                    print(f"Saved Best Model with Validation Loss: {best_val_loss}")
                
                # Save model to wandb if enabled
                if use_wandb:
                    wandb.run.summary["best_val_loss"] = best_val_loss
                    wandb.run.summary["best_epoch"] = epoch + 1
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= args.early_stopping_patience:
                    print(f"No improvement for {args.early_stopping_patience} epochs, stopping training.")
                    
                    stop_training_signal = torch.tensor(1, device=rank)
        dist.broadcast(stop_training_signal, src=0)
        if stop_training_signal.item() == 1:
            if rank != 0:
                print(f"Rank {rank} received stop training signal.")
            break


        scheduler.step()
    
    # Finish wandb run (only on rank 0 and if enabled)
    if rank == 0 and use_wandb:
        wandb.finish()
    
    cleanup_ddp()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Training Script")
    parser.add_argument('--model_type', type=str, required=True, help='Variant of the model')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Early stopping patience epochs')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to save the model checkpoint')
    parser.add_argument('--world_size', type=int, default=torch.cuda.device_count(), help='Number of GPUs to use')
    parser.add_argument('--use_wandb', type=str, default='False', help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='deep_ar_project', help='WandB project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='WandB run name (default: auto-generated)')
    parser.add_argument('--wandb_tags', type=str, default=None, help='Comma-separated tags for WandB run')
    parser.add_argument('--original_sam_checkpoint', type=str, default=None, help='Path to original SAM checkpoint for initialization')

    parser.add_argument('--pretrained_lr', type=float, default=1e-5, help='Learning rate for pre-trained parameters')
    parser.add_argument('--scratch_lr', type=float, default=1e-4, help='Learning rate for scratch parameters')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs for scratch parameters')
    parser.add_argument('--warmup_start_lr', type=float, default=1e-6, help='Starting learning rate for warmup of scratch parameters')

    parser.add_argument('--train_input_dir', type=str, required=True, help='Path to training directory containing .nc files')
    parser.add_argument('--val_input_dir', type=str, required=True, help='Path to validation directory containing .nc files')
    parser.add_argument('--train_gt_dir', type=str, required=True, help='Path to training ground truth directory containing .nc files')
    parser.add_argument('--val_gt_dir', type=str, required=True, help='Path to validation ground truth directory containing .nc files')
    
    # PEFT (Parameter-Efficient Fine-Tuning) arguments
    parser.add_argument('--peft_method', type=str, default='none', choices=['none', 'lora'],
                       help='Parameter-efficient fine-tuning method: none (full), lora')
    parser.add_argument('--lora_rank', type=int, default=8, help='LoRA rank (default: 8)')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha scaling (default: 16)')
    parser.add_argument('--lora_dropout', type=float, default=0.0, help='LoRA dropout rate (default: 0.0)')

    parser.add_argument('--use_gradient_checkpointing', type=lambda x: x.lower() == 'true', default=True, help='Use gradient checkpointing to save memory')

    args = parser.parse_args()

    mp.spawn(train_ddp,
             args=(args.world_size, args),
             nprocs=args.world_size,
             join=True)