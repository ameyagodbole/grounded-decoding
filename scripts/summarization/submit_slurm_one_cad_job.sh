#! /bin/bash
#SBATCH --job-name=cad_head_mask-fp16
#SBATCH --output=logs/ragtruth_news-llama2-chat-cad_head_mask-%A-%a.out
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a6000:1
#SBATCH --exclude=glamor-ruby
#SBATCH --cpus-per-task=4

which python

export PYTHONPATH=/home/ameya/grounded-decoding:${PYTHONPATH}
export TOKENIZERS_PARALLELISM=false

python -u scripts/summarization/run_headmask.py \
    --cad_alpha=0.5 --clamp_logit_diff=0 --mask_topk=50 \
    --model_nm llama2-chat --dataset_nm ragtruth_news \
    --decoding_algo cad_head_mask \
    --generation_config configs/topp_config.json \
    --output_dir /home/ameya/grounded-decoding/outputs/ragtruth_news/cad_head_mask/ \
    --cad_kv_sharing total \
    --head_score_file configs/retrieval_head_score/llama-2-7b-80k.json \
    --return_js_divergence \
    --fp16 --n_samples 250 --shuffle_seed 1904 \
    --topp 0.9 --temperature 1.0 --max_length 4096 --max_new_tokens 384 \
    --log_wandb
