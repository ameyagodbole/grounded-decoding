#! /bin/bash
#SBATCH --job-name=regular-fp16
#SBATCH --output=logs/ragtruth_news-llama2-chat-regular-%A-%a.out
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a6000:1
#SBATCH --exclude=glamor-ruby
#SBATCH --cpus-per-task=4

which python

export PYTHONPATH=/home/ameya/grounded-decoding:${PYTHONPATH}
export TOKENIZERS_PARALLELISM=false

case $SLURM_ARRAY_TASK_ID in
    0)
        for r in {0..2}
        do
            python -u scripts/summarization/run_headmask.py \
                --model_nm llama2-chat --dataset_nm wiki_bios \
                --decoding_algo regular \
                --generation_config configs/topp_config.json \
                --output_dir /home/ameya/grounded-decoding/outputs/wiki_bios/regular-v3/ \
                --fp16 --n_samples 180 --shuffle_seed 1904 \
                --topp 0.9 --temperature 1.0 --max_length 4096 --max_new_tokens 384 \
                --log_wandb
        done
        ;;
    1)
        for r in {0..2}
        do
            python -u scripts/summarization/run_headmask.py \
                --model_nm llama2-chat --dataset_nm cnn_dailymail \
                --decoding_algo regular \
                --generation_config configs/topp_config.json \
                --output_dir /home/ameya/grounded-decoding/outputs/cnn_dailymail/regular-v3/ \
                --fp16 --n_samples 250 --shuffle_seed 1904 \
                --topp 0.9 --temperature 1.0 --max_length 4096 --max_new_tokens 384 \
                --log_wandb
        done
        ;;
    2)
        for r in {0..2}
        do
            python -u scripts/summarization/run_headmask.py \
                --model_nm llama2-chat --dataset_nm xsum \
                --decoding_algo regular \
                --generation_config configs/topp_config.json \
                --output_dir /home/ameya/grounded-decoding/outputs/xsum/regular-v3/ \
                --fp16 --n_samples 250 --shuffle_seed 1904 \
                --topp 0.9 --temperature 1.0 --max_length 4096 --max_new_tokens 384 \
                --log_wandb
        done
        ;;
    3)
        for r in {0..2}
        do
            python -u scripts/summarization/run_headmask.py \
                --model_nm llama2-chat --dataset_nm ragtruth_news \
                --decoding_algo regular \
                --generation_config configs/topp_config.json \
                --output_dir /home/ameya/grounded-decoding/outputs/ragtruth_news/regular-v3/ \
                --fp16 --n_samples 250 --shuffle_seed 1904 \
                --topp 0.9 --temperature 1.0 --max_length 4096 --max_new_tokens 384 \
                --log_wandb
        done
        ;;
esac

# python -u scripts/summarization/run_headmask.py \
#     --model_nm llama2-chat --dataset_nm ragtruth_news \
#     --decoding_algo regular \
#     --generation_config configs/topp_config.json \
#     --output_dir /home/ameya/grounded-decoding/outputs/ragtruth_news/regular-v3/ \
#     --fp16 --n_samples 250 --shuffle_seed 1904 \
#     --topp 0.9 --temperature 1.0 --max_length 4096 --max_new_tokens 384 \
#     --log_wandb
