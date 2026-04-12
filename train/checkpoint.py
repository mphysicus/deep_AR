import json
from pathlib import Path

import torch.nn as nn
from accelerate import Accelerator
from safetensors.torch import save_file, load_file
from peft import get_peft_model_state_dict, set_peft_model_state_dict

class CheckpointManager:
    """Manages saving and loading of model checkpoints."""
    
    def __init__(self, output_dir: Path, accelerator: Accelerator, train_method: str):
        self.output_dir = output_dir
        self.accelerator = accelerator
        self.train_method = train_method
        
    def save_best_checkpoint(self, model: nn.Module, epoch: int, best_val_loss: float, train_loss: float):
        """
        Saves two distinct sets of weights:
        1. The full FSDP training state (optimizers, schedulers) for resuming training later.
        2. A lightweight 'best_model.safetensors' (only LoRA adapters & custom layers) 
           which is later loaded by final_merge_and_unload() to generate the final inference model.
        """
        self.accelerator.print(f"New best val_loss: {best_val_loss:.4f} at epoch {epoch}. Saving checkpoint...")

        checkpoint_dir = self.output_dir / "best_checkpoint"
        self.accelerator.save_state(checkpoint_dir)

        # Gather full model state_dict (collective call, must run on ALL ranks with FSDP)
        full_state_dict = self.accelerator.get_state_dict(model)

        if self.accelerator.is_main_process:
            metadata = {
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "train_loss": train_loss
            }
            with open(checkpoint_dir / "training_metadata.json", "w") as f:
                json.dump(metadata, f)
        
            save_path = self.output_dir / "best_model.safetensors"
            
            if self.train_method == 'lora':
                self.accelerator.print("Saving only LoRA adapters and trainable weights to best_model.safetensors...")
                
                # Unwrap the model to access named_parameters without FSDP wrappers
                unwrapped_model = self.accelerator.unwrap_model(model)

                state_dict_to_save = get_peft_model_state_dict(unwrapped_model, state_dict=full_state_dict)
            else:
                state_dict_to_save = full_state_dict
            
            save_file(state_dict_to_save, str(save_path))
            self.accelerator.print(f"Saved best model weights to {save_path}")

    def load_training_state(self, checkpoint_path: str):
        """Loads FSDP training state and returns the start epoch and best val loss."""
        ckpt_path = Path(checkpoint_path)
        is_full_checkpoint = (ckpt_path.is_dir() and (ckpt_path / "scheduler.bin").exists())

        start_epoch = 1
        best_val_loss = float("inf")

        if is_full_checkpoint:
            self.accelerator.print(f"Resuming full training state from: {ckpt_path}...")
            self.accelerator.load_state(ckpt_path)

            metadata_path = ckpt_path / "training_metadata.json"
            if metadata_path.exists() and self.accelerator.is_main_process:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                start_epoch = metadata.get("epoch", 0) + 1
                best_val_loss = metadata.get("best_val_loss", float("inf"))
            
            # Broadcast metadata to all ranks
            from accelerate.utils import broadcast_object_list
            metadata_list = [start_epoch, best_val_loss]
            broadcast_object_list(metadata_list, from_process=0)
            start_epoch, best_val_loss = metadata_list
            
            if self.accelerator.is_main_process:
                self.accelerator.print(f"Resuming from epoch {start_epoch}, best_val_loss: {best_val_loss:.4f}")
        else:
            self.accelerator.print(f"Note: {ckpt_path} is not a full training state checkpoint.",
                              "Assuming weights were loaded before prepare(). Starting new training loop.")

        return start_epoch, best_val_loss

    def final_merge_and_unload(self, model: nn.Module):
        """
        Merges LoRA adapter into the base model at the end of training for a pristine inference model.
        """
        if self.train_method == 'lora' and self.accelerator.is_main_process:
            self.accelerator.print("\nTraining complete. Performing merge_and_unload for the final inference checkpoint...")
            
            best_model_path = self.output_dir / "best_model.safetensors"
            
            if best_model_path.exists():
                self.accelerator.print(f"Loading best model weights from {best_model_path} for merging...")
                adapter_state = load_file(str(best_model_path))
                set_peft_model_state_dict(model, adapter_state)
                
                final_model = model.merge_and_unload()
                
                final_save_path = self.output_dir / "inference_ready_model.safetensors"
                save_file(final_model.state_dict(), str(final_save_path))
                self.accelerator.print(f"Saved ready-for-inference merged model to {final_save_path}")
            else:
                self.accelerator.print(f"Warning: {best_model_path} not found, skipping merge.")