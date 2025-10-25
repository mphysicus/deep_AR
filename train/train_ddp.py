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
from deep_ar.data.datasets import IVT_dataset

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

def train_one_epoch(model, dataloader, optimizer, criterion, rank, scaler):
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

    #Build model
    model_builder = deep_ar_model_registry[args.model_type]
    model = model_builder(checkpoint=args.checkpoint)

    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    train_dataset = IVT_dataset(args.train_tensor_input)
    val_dataset = IVT_dataset(args.val_tensor_input)

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

    best_val_loss = torch.inf

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        epoch_loss = train_one_epoch(model, train_loader, optimizer, criterion, rank, scaler)

        if rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Training Loss: {epoch_loss}")

        val_loss = validate(model, val_loader, criterion, rank)

        if rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Validation Loss: {val_loss}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.module.state_dict(), f"best_model.pth")
                print(f"Saved Best Model with Validation Loss: {best_val_loss}")

        scheduler.step()
    cleanup_ddp()

parser = argparse.ArgumentParser(description="Distributed Training Script")
parser.add_argument('--model_type', type=str, required=True, help='Variant of the model')
parser.add_argument('--train_tensor_input', type=str, required=True, help='Path to training tensor file')
parser.add_argument('--val_tensor_input', type=str, required=True, help='Path to validation tensor file')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size per GPU')
parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
parser.add_argument('--world_size', type=int, default=torch.cuda.device_count(), help='Number of GPUs to use')
args = parser.parse_args()

if __name__ == "__main__":
    mp.spawn(train_ddp,
             args=(args.world_size, args),
             nprocs=args.world_size,
             join=True)