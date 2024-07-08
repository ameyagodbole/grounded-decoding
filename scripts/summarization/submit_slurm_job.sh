#! /bin/bash
#SBATCH --job-name=wikibios-cad_head_mask-fp16-sweep-ajz0hcvn-agent
#SBATCH --output=logs/wikibios-llama2-chat-cad_head_mask-sweep-ajz0hcvn-%A-%a.out
#SBATCH --time=24:00:00
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
# wandb agent ameyag416/grounded-decoding/58u6uyd7

######
# V2 #
######

# cnn_dailymail cad_head_mask + total kv_cache_sharing + log js_divergences
# wandb agent ameyag416/grounded-decoding/we14u13i
# fp16
# wandb agent ameyag416/grounded-decoding/c9yzgor1
# max_gen_len=200
# wandb agent ameyag416/grounded-decoding/zmf4icsl
# fp16 + max_gen_len=200
# wandb agent ameyag416/grounded-decoding/zmbae8vt
# fp16 + max_gen_len=200 + clamp_logit_diff
# wandb agent ameyag416/grounded-decoding/kmnz5ksa
# cad_head_context_mask + fp16
# wandb agent ameyag416/grounded-decoding/7fqbfg5r

# xsum cad_head_mask + total kv_cache_sharing + fp16 + max_gen_len=200 + clamp_logit_diff
# wandb agent ameyag416/grounded-decoding/2j09zr2t
# cad_head_context_mask + fp16
# wandb agent ameyag416/grounded-decoding/5nnuvv19


# wiki_bios cad_head_mask + total kv_cache_sharing + log js_divergences + fp16 + max_gen_len=200
# wandb agent ameyag416/grounded-decoding/z72lgrn0
# restart with prompt that limits output length
# wandb agent ameyag416/grounded-decoding/jxbqaitn
# fp16 + max_gen_len=200 + clamp_logit_diff
# wandb agent ameyag416/grounded-decoding/e1gwe0yw
# cad_head_context_mask + fp16
# wandb agent ameyag416/grounded-decoding/bu9snjv8

######
# V3 #
######

# cnn_dailymail
# wandb agent ameyag416/grounded-decoding/o23cpsu4

# wiki_bios
wandb agent ameyag416/grounded-decoding/ajz0hcvn