from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp
from torch.amp import GradScaler, autocast
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deep_ar.build_deep_ar import deep_ar_model_registry
from deep_ar.data.datasets import ARTrainingDataset

# Import LoRA utilities
from lora.utils import (
    apply_lora_to_sam, 
    mark_only_lora_as_trainable as mark_lora_trainable,
    lora_state_dict as get_lora_state_dict
)

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
        images = batch['image'].to(rank)
        masks = batch['mask'].to(rank)

        optimizer.zero_grad()

        with autocast("cuda"):
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        total_train_loss += loss.item()

        scaler.scale(loss).backward()
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
            images = batch['image'].to(rank)
            masks = batch['mask'].to(rank)

            with autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, masks)

            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(dataloader)

    #Synchronize the validation loss across all processes
    avg_val_loss_tensor = torch.tensor(avg_val_loss, device=rank)
    dist.all_reduce(avg_val_loss_tensor, op=dist.ReduceOp.AVG)

    return avg_val_loss_tensor.item()

def train_ddp(rank, world_size, args):
    setup_ddp(rank, world_size)

    # Initialize wandb only on rank 0 and if enabled
    use_wandb = args.use_wandb and WANDB_AVAILABLE
    if rank == 0 and use_wandb:
        config_dict = {
            "model_type": args.model_type,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "world_size": world_size,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "peft_method": args.peft_method,
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
    model = model_builder(checkpoint=None)

    if args.checkpoint is not None:
        if rank == 0:
            print(f"Loading SAM's image encoder and mask decoder checkpoint from {args.checkpoint}...")
        
        state_dict = torch.load(args.checkpoint, map_location=f'cuda:{rank}')
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
        
        # Apply LoRA to the SAM model inside DeepAR
        apply_lora_to_sam(
            model.sam_model,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            verbose=(rank == 0)
        )
        
        # Freeze base model, only train LoRA parameters
        mark_lora_trainable(model)
        
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
    model = DDP(model, device_ids=[rank])

    train_dataset = ARTrainingDataset(args.train_tensor_input)
    val_dataset = ARTrainingDataset(args.val_tensor_input)

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
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= args.epochs)
    scaler = GradScaler()

    # Watch model (only on rank 0 and if wandb enabled)
    if rank == 0 and use_wandb:
        wandb.watch(model, log="all", log_freq=100)

    best_val_loss = torch.inf
    no_improve_epochs = 0

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        epoch_loss = train_one_epoch(model, train_loader, optimizer, criterion, rank, scaler, epoch, use_wandb)

        if rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Training Loss: {epoch_loss}")

        val_loss = validate(model, val_loader, criterion, rank)

        if rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Validation Loss: {val_loss}")
            
            # Log epoch-level metrics to wandb if enabled
            if use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": epoch_loss,
                    "val_loss": val_loss,
                    "learning_rate": scheduler.get_last_lr()[0]
                })
           
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_epochs = 0
                checkpoint_path = f"best_model_epoch_{epoch+1}.pth"
                
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
                    break

        scheduler.step()
    
    # Finish wandb run (only on rank 0 and if enabled)
    if rank == 0 and use_wandb:
        wandb.finish()
    
    cleanup_ddp()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Training Script")
    parser.add_argument('--model_type', type=str, required=True, help='Variant of the model')
    parser.add_argument('--train_tensor_input', type=str, required=True, help='Path to training tensor file')
    parser.add_argument('--val_tensor_input', type=str, required=True, help='Path to validation tensor file')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Early stopping patience epochs')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--world_size', type=int, default=torch.cuda.device_count(), help='Number of GPUs to use')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='deep_ar_project', help='WandB project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='WandB run name (default: auto-generated)')
    parser.add_argument('--wandb_tags', type=str, default=None, help='Comma-separated tags for WandB run')
    
    # PEFT (Parameter-Efficient Fine-Tuning) arguments
    parser.add_argument('--peft_method', type=str, default='none', choices=['none', 'lora'],
                       help='Parameter-efficient fine-tuning method: none (full), lora')
    parser.add_argument('--lora_rank', type=int, default=8, help='LoRA rank (default: 8)')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha scaling (default: 16)')
    parser.add_argument('--lora_dropout', type=float, default=0.0, help='LoRA dropout rate (default: 0.0)')

    args = parser.parse_args()

    mp.spawn(train_ddp,
             args=(args.world_size, args),
             nprocs=args.world_size,
             join=True)