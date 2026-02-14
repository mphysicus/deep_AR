import argparse
from pathlib import Path
from typing import Optional
from tqdm import tqdm

import torch
import torch.nn as nn
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig

from safetensors.torch import save_file, load_file
from peft import LoraConfig, get_peft_model

import wandb
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deep_ar.build_deep_ar import deep_ar_model_registry
from deep_ar.data.datasets import ARTrainingDataset
from loss import CombinedLoss
from scheduler import get_scheduler

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train DeepAR model.")
    parser.add_argument('--model_type', type=str, default='vit_b', help='Variant of the Model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size per device.")
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--min_delta', type=float, default=0.001, help='Minimum improvement for early stopping')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Early stopping patience epochs')
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Directory for checkpoints and logs")
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
    parser.add_argument('--train_method', type=str, default='none', choices=['none', 'lora', 'only_cnn'])
    parser.add_argument('--lora_rank', type=int, default=8, help='LoRA rank (default: 8)')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha scaling (default: 16)')
    parser.add_argument('--lora_dropout', type=float, default=0.0, help='LoRA dropout rate (default: 0.0)')
    parser.add_argument('--lora_target_modules', type=str, nargs='+', default=['qkv', 'proj'], help='Target modules for LoRA (default: qkv proj)')

    parser.add_argument("--wandb_project", type=str, default="deep_ar_training", help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Weights & Biases run name")

    # TODO: Experimental. Need to test if this works properly.
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint directory or .safetensors file to load before training starts")

    return parser.parse_args()

def train_one_epoch(
        criterion,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        accelerator: Accelerator,
        epoch: int,
        args: argparse.Namespace,
        scheduler = None,
        progress_bar: Optional[tqdm] = None,
) -> float:
    """Train for one epoch.

    Args:
        model: DeepAR model
        train_loader: DataLoader for training data
        optimizer: Optimizer for training
        accelerator: Accelerator for mixed precision and distributed training
        epoch: Current epoch number
        args: argparse Namespace
        scheduler: Learning rate scheduler
        progress_bar: Global tqdm progress bar
    
    Returns:
        Average training loss for the epoch
    """
    model.train()
    local_total_loss = 0.0
    local_num_batches = 0
    accumulated_loss = 0.0 # Track loss across gradient accumulation steps

    current_step_accum = 0

    iterator = train_loader
    for step, batch in enumerate(iterator):
        with accelerator.accumulate(model):
            with accelerator.autocast():
                loss = criterion(model(batch['image'])['output'], batch['gt_mask'])

            # Backward pass
            accelerator.backward(loss)

            # Accumulate loss for logging
            accumulated_loss += loss.detach()
            current_step_accum += 1

            #TODO: Check this if this really helps. Need to do more research
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping

            optimizer.step()

            if accelerator.sync_gradients and scheduler is not None:
                scheduler.step()
            
            optimizer.zero_grad()

            if accelerator.sync_gradients:
                # Average loss over gradient accumulation steps
                avg_loss = accumulated_loss.item() / current_step_accum
                accumulated_loss = 0.0 # Reset for next accumulation cycle
                current_step_accum = 0

                local_total_loss += avg_loss
                local_num_batches += 1

            # Update global progress bar
                if progress_bar is not None:
                    progress_bar.set_postfix({"loss": f"{avg_loss:.4f}", "epoch": epoch}, refresh=False)
                    progress_bar.update(1)

            # Log to wandb only on actual gradient update steps
                if accelerator.is_main_process:
                    # Use progress bar position as global_step (already accounts for resuming)
                    global_step = progress_bar.n if progress_bar is not None else 0
                    
                    log_dict = {"train/step_loss": avg_loss}
                    if scheduler is not None:
                        log_dict["train/lr"] = scheduler.get_last_lr()[0]

                    accelerator.log(log_dict, step=global_step)

    # Global reduction at the end
    # Convert to tensors for communication
    total_loss_tensor = torch.tensor(local_total_loss, device=accelerator.device)
    total_batches_tensor = torch.tensor(local_num_batches, device=accelerator.device)

    # Sum up the loss and batch count across all processes
    global_total_loss = accelerator.reduce(total_loss_tensor, reduction="sum").item()
    global_total_batches = accelerator.reduce(total_batches_tensor, reduction="sum").item()

    # Calculate the true Global Average
    if global_total_batches > 0:
        return global_total_loss / global_total_batches
    else:
        return 0.0
    
@torch.no_grad()
def validate(
    criterion,
    model: nn.Module,
    val_loader: DataLoader,
    accelerator: Accelerator,
) -> float:
    """
    Validate the model.

    Args:
        model: DeepAR model to train
        val_loader: DataLoader for validation data
        accelerator: Accelerator instance

    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(
        val_loader,
        desc="Validating",
        disable=not accelerator.is_local_main_process
    ):
        with accelerator.autocast():
            loss = criterion(model(batch['image'])['output'], batch['gt_mask'])

        # Accumulate loss on local device first, gather once at the end
        total_loss += loss.item()
        num_batches += 1

    # Single all-reduce at the end instead of per-batch
    total_loss = torch.tensor(total_loss, device=accelerator.device)
    num_batches = torch.tensor(num_batches, device=accelerator.device)

    total_loss = accelerator.reduce(total_loss, reduction="sum").item()
    num_batches = accelerator.reduce(num_batches, reduction="sum").item()

    return total_loss / num_batches if num_batches > 0 else 0.0

def train(
        criterion,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        accelerator: Accelerator,
        max_epochs: int,
        early_stopping_patience: int,
        min_delta: float,
        output_dir: Path,
        args: argparse.Namespace,
        start_epoch: int = 1,
        best_val_loss: float = float("inf"),
        progress_bar: Optional[tqdm] = None,
):
    """
    Train the model.

    Args:
        model: DeepAR model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler
        accelerator: Accelerator for mixed precision and distributed training
        max_epochs: Maximum number of epochs to train
        early_stopping_patience: Patience for early stopping
        min_delta: Minimum improvement to reset patience
        output_dir: Directory to save checkpoints
        args: Command line arguments
        start_epoch: Epoch to start/resume training from
        best_val_loss: Best validation loss from previous training (for resuming)
        progress_bar: Global tqdm progress bar
    """
    epochs_without_improvement = 0
    best_epoch = start_epoch - 1

    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, max_epochs + 1):
        train_loss = train_one_epoch(criterion,
            model, train_loader, optimizer, accelerator,
            epoch, args, scheduler, progress_bar
        )
        val_loss = validate(
            criterion, model, val_loader, accelerator
        )

        # Early stopping logic
        if val_loss + min_delta < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            accelerator.print(f"New best val_loss: {best_val_loss:.4f} at epoch {best_epoch}. Saving checkpoint...")

            # Save full training state using accelerate
            checkpoint_dir = output_dir / "best_checkpoint"
            accelerator.save_state(checkpoint_dir)

            # Gather full model state_dict (collective call, must run on ALL ranks with FSDP)
            full_state_dict = accelerator.get_state_dict(model)

            # Save additional training metadata and standalone weights
            if accelerator.is_main_process:
                import json
                metadata = {
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "train_loss": train_loss
                }
                with open(checkpoint_dir / "training_metadata.json", "w") as f:
                    json.dump(metadata, f)
            
            if accelerator.is_main_process:
                save_path = output_dir / f"best_model.safetensors"
                save_file(full_state_dict, str(save_path))
                accelerator.print(f"Saved best model weights to {save_path}")

        else:
            epochs_without_improvement += 1
            if accelerator.is_main_process:
                accelerator.print(f"No improvement in val_loss for {epochs_without_improvement} epoch(s). Current val_loss: {val_loss:.4f}, Best val_loss: {best_val_loss:.4f}")

        
        if accelerator.is_main_process:
            epoch_step = progress_bar.n if progress_bar is not None else 0
            accelerator.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "epoch": epoch,
            }, step=epoch_step)

            accelerator.print(f"Epoch: {epoch} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        # Check for early stopping
        if epochs_without_improvement >= early_stopping_patience:
            if accelerator.is_main_process:
                accelerator.print(f"Early stopping triggered after {epoch} epochs. Best val_loss: {best_val_loss:.4f} at epoch {best_epoch}.")
            break
    
    if progress_bar is not None:
        progress_bar.close()

def main():
    args = parse_args()

    set_seed(args.seed)

    output_dir = Path(args.output_dir)

    fsdp_plugin_kwargs = {
        "fsdp_version": 2,
        "state_dict_config": FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        "limit_all_gathers": True,
        "reshard_after_forward": True,  # Required for FSDP2
        "auto_wrap_policy": "transformer_based_wrap",
        "transformer_cls_names_to_wrap": ["Block", "TwoWayAttentionBlock"],
        "mixed_precision_policy": "bf16"
    }
    fsdp_plugin = FullyShardedDataParallelPlugin(**fsdp_plugin_kwargs)

    # Initialize Accelerator
    accelerator = Accelerator(
        fsdp_plugin=fsdp_plugin,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with="wandb",
    )
    accelerator.print(f"Number of GPUs detected: {accelerator.num_processes}")

    # Initialize wandb and determine run name
    run_name = None
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config=vars(args),
            init_kwargs={"wandb": {"name": args.wandb_run_name}}
        )

        # Get run name from wandb or fallback
        if wandb.run and wandb.run.name:
            run_name = wandb.run.name
        else:
            run_name = args.wandb_run_name if args.wandb_run_name else "run_unknown"
    
    # Broadcast run_name to all processes so output_dir is consistent
    from accelerate.utils import broadcast_object_list
    run_name_list = [run_name]
    broadcast_object_list(run_name_list, from_process=0)
    run_name = run_name_list[0]
    
    # Now all ranks have the same output_dir
    output_dir = output_dir / run_name
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    accelerator.print(f"Using devices: {accelerator.device}")
    accelerator.print(f"Number of processes: {accelerator.num_processes}")

    accelerator.print(f"Loading model {args.model_type}")
    model_builder = deep_ar_model_registry[args.model_type]
    model = model_builder(checkpoint=None)

    # Load original SAM checkpoint on CPU (rank 0 only, sync_module_states will broadcast)
    if args.original_sam_checkpoint is not None:
        if accelerator.is_main_process:
            accelerator.print(f"Loading original SAM checkpoint from {args.original_sam_checkpoint} on CPU...")
            state_dict = torch.load(args.original_sam_checkpoint, map_location="cpu", weights_only=True)
            adapted_state_dict = {"sam_model." + k: v for k, v in state_dict.items()}
            load_result = model.load_state_dict(adapted_state_dict, strict=False)
            accelerator.print(f"Successfully loaded SAM weights. FSDP2 will shard and move to GPU during prepare().")
            if load_result.missing_keys:
                accelerator.print(f"Missing keys: {load_result.missing_keys}")
            if load_result.unexpected_keys:
                accelerator.print(f"Unexpected keys: {load_result.unexpected_keys}")
        accelerator.wait_for_everyone()

    if args.train_method == 'lora':
        accelerator.print(f"Applying LoRA PEFT with rank {args.lora_rank}, alpha {args.lora_alpha}, dropout {args.lora_dropout}")
        lora_config = LoraConfig(
            r = args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            bias="none",
            task_type=None,
            modules_to_save=["no_mask_embedding"],
        )
        model.sam_model = get_peft_model(model.sam_model, lora_config)
        if accelerator.is_main_process:
            model.sam_model.print_trainable_parameters()
    
    elif args.train_method == 'only_cnn':
        accelerator.print(f"Training only the new CNN module (IVT2RGB).")

        for param in model.sam_model.parameters():
            param.requires_grad = False
        
        if hasattr(model.sam_model, 'no_mask_embedding'):
            model.sam_model.no_mask_embedding.requires_grad = True
        
        if accelerator.is_main_process:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            accelerator.print(f"Total parameters: {total_params}, Trainable parameters: {trainable_params} ({100 * trainable_params / total_params:.2f}%)")

            for name, param in model.named_parameters():
                if param.requires_grad:
                    accelerator.print(f"Trainable parameter: {name} with shape {param.shape}")

    elif args.train_method == 'none':
        accelerator.print(f"Training all parameters (no PEFT).")
    
    else:
        raise ValueError(f"Unsupported PEFT method: {args.train_method}")
    
    # Build parameter groups with different learning rates
    params_scratch = []
    params_pretrained = []

    scratch_param_keywords = [
        'patch_embed',
        'no_mask_embedding',
        'input_generator'
    ]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # Skip frozen parameters

        is_scratch = any(keyword in name for keyword in scratch_param_keywords)

        if is_scratch:
            params_scratch.append(param)
        else:
            params_pretrained.append(param)

    accelerator.print(f"Found {len(params_pretrained)} 'pre-trained' parameter tensors.")
    accelerator.print(f"Found {len(params_scratch)} 'from scratch' parameter tensors.")

    param_groups = [
        {"params": params_pretrained, "lr": args.pretrained_lr, 'name': 'pretrained'},
        {"params": params_scratch, "lr": args.scratch_lr, 'name': 'scratch'},
    ]

    # Create datasets and dataloaders
    train_input_files = sorted(list(Path(args.train_input_dir).glob("*.nc")))
    val_input_files = sorted(list(Path(args.val_input_dir).glob("*.nc")))
    train_gt_files = sorted(list(Path(args.train_gt_dir).glob("*.nc")))
    val_gt_files = sorted(list(Path(args.val_gt_dir).glob("*.nc")))

    train_dataset = ARTrainingDataset(input_files=train_input_files, gt_files=train_gt_files)
    val_dataset = ARTrainingDataset(input_files=val_input_files, gt_files=val_gt_files)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = get_scheduler(optimizer, args)

    # Load safetensors checkpoint BEFORE prepare() on rank 0 only (if provided)
    if args.resume_from_checkpoint and args.resume_from_checkpoint.endswith(".safetensors") and accelerator.is_main_process:
        accelerator.print(f"Loading weights from {args.resume_from_checkpoint} on rank 0...")
        state = load_file(args.resume_from_checkpoint)
        model.load_state_dict(state, strict=False)
        accelerator.print(f"Successfully loaded weights on rank 0. Will sync to all ranks during prepare().")

    accelerator.print("Preparing model, optimizer, and dataloaders with Accelerator...")
    # Prepare everything with accelerator
    # sync_module_states=True will broadcast weights from rank 0 to all ranks
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    num_batches_per_epoch = len(train_loader)
    num_update_steps_per_epoch = (num_batches_per_epoch + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps
    max_train_steps = args.epochs * num_update_steps_per_epoch

    start_epoch = 1
    best_val_loss = float("inf")

    if args.resume_from_checkpoint is not None:
        accelerator.print(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")

        checkpoint_path = Path(args.resume_from_checkpoint)

        is_full_checkpoint = (checkpoint_path.is_dir() and (checkpoint_path / "scheduler.bin").exists())

        if is_full_checkpoint:
            accelerator.print(f"Resuming full training state from: {checkpoint_path}...")

            # Load the optimizer states, scheduler states, and any other training state information
            accelerator.load_state(checkpoint_path)

            # Load training metadata
            metadata_path = checkpoint_path / "training_metadata.json"
            if metadata_path.exists():
                import json
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                start_epoch = metadata.get("epoch", 0) + 1
                best_val_loss = metadata.get("best_val_loss", float("inf"))
                accelerator.print(f"Resuming from epoch {start_epoch}, best_val_loss: {best_val_loss:.4f}")
        else:
            accelerator.print(f"Note: {checkpoint_path} is not a full training state checkpoint.",
                              "Assuming weights were loaded before prepare(). Starting new training loop.")

    # Initialize global progress bar AFTER loading checkpoint to get correct start_epoch
    # Calculate how many steps have already been completed
    steps_completed = (start_epoch - 1) * num_update_steps_per_epoch
    global_progress_bar = tqdm(
        total=max_train_steps,
        initial=steps_completed,
        disable=not accelerator.is_local_main_process,
        desc="Training Steps"
    )

    # Start training
    accelerator.print("Starting training...")
    accelerator.print(f"Total epochs: {args.epochs}")
    accelerator.print(f"Start epoch: {start_epoch}")
    accelerator.print(f"Batch size per device: {args.batch_size}")
    accelerator.print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    accelerator.print(f"Effective batch size: {args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")
    accelerator.print(f"Total training steps: {max_train_steps}")

    train(
        criterion=criterion,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        accelerator=accelerator,
        max_epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        min_delta=args.min_delta,
        output_dir=output_dir,
        args=args,
        start_epoch=start_epoch,
        best_val_loss=best_val_loss,
        progress_bar=global_progress_bar
    )

    accelerator.end_training()
    accelerator.print("Training complete.")

if __name__ == "__main__":
    main()