#!/bin/bash

echo "--- [launch.sh] Starting new DDP trial at $(date) ---"
echo "[launch.sh] Hyperparameters received: $@"

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(shuf -i 20000-65000 -n 1)
export RDZV_ID="${SLURM_JOB_ID}_$(date +%s)"

echo "[launch.sh] Master: $MASTER_ADDR, Port: $MASTER_PORT, ID: $RDZV_ID"

CMD="source /home/prakhar/miniconda3/bin/activate semester; \
     torchrun \
        --nnodes=${SLURM_NNODES} \
        --nproc-per-node=3 \
        --rdzv_id=$RDZV_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        --rdzv-conf="read_timeout=300" \
        train_ddp_multinode.py $@"


srun --export=ALL,MASTER_ADDR,MASTER_PORT,RDZV_ID \
  --nodes=${SLURM_NNODES} \
  --ntasks-per-node=1 \
  --cpus-per-task=${SLURM_CPUS_PER_TASK} \
  --gres=gpu:3 \
  bash -c "$CMD"

echo "--- [launch.sh] Trial finished at $(date) ---"