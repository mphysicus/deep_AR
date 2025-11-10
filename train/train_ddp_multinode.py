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
from datetime import timedelta

from scheduler import get_scheduler

# Import LoRA utilities
from lora.utils import apply_lora_to_sam

from adalora.utils import convert_linear_to_adalora, get_adalora_internal_metrics
from adalora.adalora import RankAllocator, compute_orth_regu

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def setup_ddp():
    dist.init_process_group("nccl", init_method="env://", timeout=timedelta(minutes=30))
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    torch.cuda.set_device(local_rank)
    if rank == 0:
        print(f"DDP initialized: world_size={world_size}, local_rank={local_rank}, rank={rank}")
    return rank, world_size, local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def train_one_epoch(model, dataloader, optimizer, criterion, local_rank, rank, scaler, epoch, use_wandb, rank_allocator, args):
    model.train()
    total_train_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        global_step = epoch * len(dataloader) + batch_idx

        images = batch['image'].to(local_rank, non_blocking=True)
        masks = batch['gt_mask'].to(local_rank, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda"):
            outputs = model(images)
            logits = outputs['output']
            loss = criterion(logits, masks)

            if args.peft_method == 'adalora' and rank_allocator is not None:
                regu_loss = compute_orth_regu(model.module, args.adalora_orth_reg_weight)
                loss = loss + regu_loss

        total_train_loss += loss.item()

        scaler.scale(loss).backward()

        if args.peft_method == 'adalora' and rank_allocator is not None:
            scaler.unscale_(optimizer)
            rank_allocator.update_and_mask(model.module, global_step)
        
        del outputs, logits
        scaler.step(optimizer)
        scaler.update()


        if rank==0 and batch_idx % 10 == 0:
            print(f"Rank {rank}, Batch {batch_idx}, Loss: {loss.item()}")
            
            # Log batch-level metrics to wandb if enabled
            if use_wandb:
                log_dict = {
                    "batch_loss": loss.item(),
                    "batch": global_step
                }
                if args.peft_method == 'adalora':
                    if 'regu_loss' in locals():
                        log_dict["adalora_regu_loss_avg"] = regu_loss.item()
                    
                    internal_metrics = get_adalora_internal_metrics(model.module)
                    log_dict.update(internal_metrics)

                wandb.log(log_dict)

    return total_train_loss / len(dataloader)

def validate(model, dataloader, criterion, local_rank, rank):
    """
    Validate the model on the validation dataset.
    """
    model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(local_rank, non_blocking=True)
            masks = batch['gt_mask'].to(local_rank, non_blocking=True)

            with autocast("cuda"):
                outputs = model(images)
                logits = outputs['output']
                loss = criterion(logits, masks)

            total_val_loss += loss.item()

            del outputs, logits, loss

    avg_val_loss = total_val_loss / len(dataloader)

    #Synchronize the validation loss across all processes
    avg_val_loss_tensor = torch.tensor(avg_val_loss, device=local_rank)
    dist.all_reduce(avg_val_loss_tensor, op=dist.ReduceOp.AVG)

    return avg_val_loss_tensor.item()

def train_ddp(args):
    rank, world_size, local_rank = setup_ddp()

    if args.batch_size % world_size != 0:
        if rank == 0:
            print(f"Warning: Global batch size {args.batch_size} is not divisible by world size {world_size}.")
            print("Rounding down to nearest divisible batch size.")
    
    per_gpu_batch_size = args.batch_size // world_size

    if per_gpu_batch_size == 0:
        raise ValueError(f"Per-GPU batch size is zero. Increase global batch size {args.batch_size} or reduce world size {world_size}.")
    
    if rank == 0:
        print(f"--- Batch Size ---")
        print(f"  Global batch size: {args.batch_size}")
        print(f"  World size (GPUs): {world_size}")
        print(f"  Calculated per-GPU batch size: {per_gpu_batch_size}")
        print(f"------------------")

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
        elif args.peft_method == 'adalora':
            config_dict.update({
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "adalora_init_r": args.adalora_init_r,
                "adalora_target_r": args.adalora_target_r,
                "adalora_warmup_steps": args.adalora_warmup_steps,
                "adalora_update_interval": args.adalora_update_interval,
                "adalora_orth_reg_weight": args.adalora_orth_reg_weight,
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
        
        state_dict = torch.load(args.original_sam_checkpoint, map_location=f'cuda:{local_rank}')
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
    
    elif args.peft_method == 'adalora':
        if rank == 0:
            print(f"Applying AdaLoRA (init_r={args.adalora_init_r}, target_r={args.adalora_target_r}, alpha={args.lora_alpha})")

        for param in model.sam_model.parameters():
            param.requires_grad = False
        
        ADALORA_TARGET_MODULES = [
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

        convert_linear_to_adalora(
            model.sam_model,
            r = args.adalora_init_r,
            lora_alpha = args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=ADALORA_TARGET_MODULES
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

    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

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
        batch_size=per_gpu_batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=per_gpu_batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(param_groups)

    rank_allocator = None
    if args.peft_method == 'adalora':
        total_steps = len(train_loader) * args.epochs
        if rank == 0:
            print(f"Initializing AdaLoRA RankAllocator....")
            print(f"  Total training steps: {total_steps}")
            print(f"  Initial warmup steps: {args.adalora_warmup_steps}")
            print(f"  Final warmup steps: {args.adalora_final_warmup_steps}")
            print(f"  Rank update interval: {args.adalora_update_interval} steps")

        rank_allocator = RankAllocator(model.module,
                                       lora_r=args.adalora_init_r,
                                       target_rank = args.adalora_target_r,
                                       init_warmup=args.adalora_warmup_steps,
                                       final_warmup = args.adalora_final_warmup_steps,
                                       mask_interval=args.adalora_update_interval,
                                       beta1 = args.adalora_beta1,
                                       beta2 = args.adalora_beta2,
                                       total_step=total_steps)
    
    scheduler = get_scheduler(optimizer, args)
    
    scaler = GradScaler()

    # Watch model (only on rank 0 and if wandb enabled)
    if rank == 0 and use_wandb:
        wandb.watch(model, log="all", log_freq=100)
    
    if rank == 0 and args.checkpoint:
        os.makedirs(args.checkpoint, exist_ok=True)

    best_val_loss = torch.inf
    no_improve_epochs = 0

    stop_training_signal = torch.tensor(0, device=local_rank)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)

        current_lr_pretrained = optimizer.param_groups[0]['lr']
        current_lr_scratch = optimizer.param_groups[1]['lr']

        epoch_loss = train_one_epoch(model, train_loader, optimizer, criterion, 
                                     local_rank, rank, scaler, epoch, use_wandb,
                                     rank_allocator, args)

        if rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Training Loss: {epoch_loss}")
            print(f"  Learning Rates - Pretrained: {current_lr_pretrained}, Scratch: {current_lr_scratch}")

        val_loss = validate(model, val_loader, criterion, 
                            local_rank, rank)

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
                checkpoint_path = f"{args.checkpoint}/best_model_epoch_{epoch+1}_batch{args.batch_size}_val_loss{val_loss}.pth"
                torch.save(model.module.state_dict(), checkpoint_path)
                
                # Save model to wandb if enabled
                if use_wandb:
                    wandb.run.summary["best_val_loss"] = best_val_loss
                    wandb.run.summary["best_epoch"] = epoch + 1
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= args.early_stopping_patience:
                    print(f"No improvement for {args.early_stopping_patience} epochs, stopping training.")
                    stop_training_signal.fill_(1)

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
    parser.add_argument('--batch_size', type=int, default=8, help='Global batch size across all GPUs')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Early stopping patience epochs')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to save the model checkpoint')
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
    parser.add_argument('--peft_method', type=str, default='none', choices=['none', 'lora', 'adalora'],
                       help='Parameter-efficient fine-tuning method: none (full), lora, adalora')
    parser.add_argument('--lora_rank', type=int, default=32, help='LoRA rank (default: 32)')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha scaling (default: 32)')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout rate (default: 0.1)')

    parser.add_argument('--adalora_init_r', type=int, default=12, help='AdaLoRA initial rank (if peft_method=adalora)')
    parser.add_argument('--adalora_target_r', type=int, default=8, help='AdaLoRA target average rank (if peft_method=adalora)')
    parser.add_argument('--adalora_warmup_steps', type=int, default=1000, help='AdaLoRA initial warmup steps (if peft_method=adalora)')
    parser.add_argument('--adalora_final_warmup_steps', type=int, default=0, help='AdaLoRA final warmup steps before end of training')
    parser.add_argument('--adalora_update_interval', type=int, default=10, help='AdaLoRA rank update interval (in steps)')
    parser.add_argument('--adalora_beta1', type=float, default=0.85, help='AdaLoRA beta1 for sensitivity EMA')
    parser.add_argument('--adalora_beta2', type=float, default=0.85, help='AdaLoRA beta2 for uncertainty EMA')
    parser.add_argument('--adalora_orth_reg_weight', type=float, default=0.1, help='AdaLoRA orthogonal regularization weight')

    parser.add_argument('--use_gradient_checkpointing', type=lambda x: x.lower() == 'true', default=True, help='Use gradient checkpointing to save memory')

    args = parser.parse_args()

    train_ddp(args)