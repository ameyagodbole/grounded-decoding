#! /bin/bash
#SBATCH --job-name=xsum-self_cad_all_token-sweep-6d9mya0x-agent-4
#SBATCH --output=logs/xsum-llama2-chat-self_cad_all_token-sweep-6d9mya0x-%A.out
#SBATCH --time=15:00:00
#SBATCH --gres=gpu:a6000:1
#SBATCH --exclude=glamor-ruby,allegro-adams
#SBATCH --cpus-per-task=1

which python

export PYTHONPATH=/home/ameya/grounded-decoding:${PYTHONPATH}
wandb agent ameyag416/grounded-decoding/6d9mya0x