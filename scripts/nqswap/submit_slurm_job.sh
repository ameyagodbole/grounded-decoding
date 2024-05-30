#! /bin/bash
#SBATCH --job-name=nq-cad_head_mask-sweep-k2662be1-agent
#SBATCH --output=logs/nq-llama2-chat-cad_head_mask-sweep-k2662be1-%A-%a.out
#SBATCH --time=15:00:00
#SBATCH --gres=gpu:a6000:1
#SBATCH --exclude=glamor-ruby
#SBATCH --cpus-per-task=1

which python

export PYTHONPATH=/home/ameya/grounded-decoding:${PYTHONPATH}
export TOKENIZERS_PARALLELISM=false

# v1.0 sweep
# wandb agent ameyag416/grounded-decoding/stv9e81k

# NQ-Swap head-mask
# wandb agent ameyag416/grounded-decoding/xwmgk1w5

# NQ head-mask
wandb agent ameyag416/grounded-decoding/k2662be1