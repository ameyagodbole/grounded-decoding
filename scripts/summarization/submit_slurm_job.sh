#! /bin/bash
#SBATCH --job-name=xsum-cad_head_mask-sweep-58u6uyd7-agent
#SBATCH --output=logs/xsum-llama2-chat-cad_head_mask-total_kv_sharing-sweep-58u6uyd7-%A-%a.out
#SBATCH --time=15:00:00
#SBATCH --gres=gpu:a6000:1
#SBATCH --exclude=glamor-ruby
#SBATCH --cpus-per-task=1

which python

export PYTHONPATH=/home/ameya/grounded-decoding:${PYTHONPATH}
export TOKENIZERS_PARALLELISM=false

# xsum self_cad_all_token
# wandb agent ameyag416/grounded-decoding/6d9mya0x

# xsum self_cad + total kv_cache_sharing
# wandb agent ameyag416/grounded-decoding/cotf9zg0

# xsum (random beta) self_cad + total kv_cache_sharing
# wandb agent --count 20 ameyag416/grounded-decoding/qlhrzwpo

# xsum (random beta) self_cad + total kv_cache_sharing + log js_divergences
# wandb agent --count 20 ameyag416/grounded-decoding/ieskbokn

# xsum cad_head_mask + total kv_cache_sharing + log js_divergences
# wandb agent --count 20 ameyag416/grounded-decoding/gmqqc4u6
wandb agent ameyag416/grounded-decoding/58u6uyd7