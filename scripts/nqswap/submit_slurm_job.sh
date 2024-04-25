#! /bin/bash
#SBATCH --job-name=nqswap-self_cad_all_token-sweep-stv9e81k-agent-3
#SBATCH --output=logs/nqswap-llama2-chat-self_cad_all_token-sweep-stv9e81k-%A.out
#SBATCH --time=15:00:00
#SBATCH --gres=gpu:a6000:1
#SBATCH --exclude=glamor-ruby
#SBATCH --cpus-per-task=1

which python

export PYTHONPATH=/home/ameya/grounded-decoding:${PYTHONPATH}
wandb agent ameyag416/grounded-decoding/stv9e81k