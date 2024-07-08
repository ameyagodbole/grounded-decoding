#! /bin/bash
#SBATCH --job-name=viz-cad_head_mask-fp16
#SBATCH --output=logs/viz-llama2-chat-cad_head_mask-%A-%a.out
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a6000:1
#SBATCH --exclude=glamor-ruby
#SBATCH --cpus-per-task=4

which python

export PYTHONPATH=/home/ameya/grounded-decoding:${PYTHONPATH}
export TOKENIZERS_PARALLELISM=false

case $SLURM_ARRAY_TASK_ID in
    0)
        python -u scripts/summarization/run_headmask.py \
            --cad_alpha=0.5 --cad_beta=1e-07 --clamp_logit_diff=0 --mask_topk=50 \
            --model_nm llama2-chat --dataset_nm wiki_bios \
            --decoding_algo cad_head_context_mask \
            --generation_config configs/topp_config.json \
            --output_dir /home/ameya/grounded-decoding/outputs/viz/wiki_bios/ \
            --cad_kv_sharing total \
            --head_score_file configs/retrieval_head_score/llama-2-7b-80k.json \
            --return_js_divergence --output_scores --output_logits --output_attentions --output_hidden_states \
            --fp16 --n_samples 30 --shuffle_seed 1306 --topp 0.9 --temperature 1.0 --max_new_tokens 384
        
        python -u scripts/summarization/run_headmask.py \
            --cad_alpha=0.5 --clamp_logit_diff=0 --mask_topk=50 \
            --model_nm llama2-chat --dataset_nm wiki_bios \
            --decoding_algo cad_head_mask \
            --generation_config configs/topp_config.json \
            --output_dir /home/ameya/grounded-decoding/outputs/viz/wiki_bios/ \
            --cad_kv_sharing total \
            --head_score_file configs/retrieval_head_score/llama-2-7b-80k.json \
            --return_js_divergence --output_scores --output_logits --output_attentions --output_hidden_states \
            --fp16 --n_samples 30 --shuffle_seed 1306 --topp 0.9 --temperature 1.0 --max_new_tokens 384
        ;;
    1)
        python -u scripts/summarization/run_headmask.py \
            --cad_alpha=0.5 --cad_beta=1e-07 --clamp_logit_diff=0 --mask_topk=50 \
            --model_nm llama2-chat --dataset_nm cnn_dailymail \
            --decoding_algo cad_head_context_mask \
            --generation_config configs/topp_config.json \
            --output_dir /home/ameya/grounded-decoding/outputs/viz/cnn_dailymail/ \
            --cad_kv_sharing total \
            --head_score_file configs/retrieval_head_score/llama-2-7b-80k.json \
            --return_js_divergence --output_scores --output_logits --output_attentions --output_hidden_states \
            --fp16 --n_samples 30 --shuffle_seed 1306 --topp 0.9 --temperature 1.0 --max_new_tokens 384

        python -u scripts/summarization/run_headmask.py \
            --cad_alpha=0.5 --clamp_logit_diff=0 --mask_topk=50 \
            --model_nm llama2-chat --dataset_nm cnn_dailymail \
            --decoding_algo cad_head_mask \
            --generation_config configs/topp_config.json \
            --output_dir /home/ameya/grounded-decoding/outputs/viz/cnn_dailymail/ \
            --cad_kv_sharing total \
            --head_score_file configs/retrieval_head_score/llama-2-7b-80k.json \
            --return_js_divergence --output_scores --output_logits --output_attentions --output_hidden_states \
            --fp16 --n_samples 30 --shuffle_seed 1306 --topp 0.9 --temperature 1.0 --max_new_tokens 384
        ;;
    2)
        python -u scripts/summarization/run_headmask.py \
            --cad_alpha=0.5 --cad_beta=1e-07 --clamp_logit_diff=0 --mask_topk=50 \
            --model_nm llama2-chat --dataset_nm xsum \
            --decoding_algo cad_head_context_mask \
            --generation_config configs/topp_config.json \
            --output_dir /home/ameya/grounded-decoding/outputs/viz/xsum/ \
            --cad_kv_sharing total \
            --head_score_file configs/retrieval_head_score/llama-2-7b-80k.json \
            --return_js_divergence --output_scores --output_logits --output_attentions --output_hidden_states \
            --fp16 --n_samples 30 --shuffle_seed 1306 --topp 0.9 --temperature 1.0 --max_new_tokens 384

        python -u scripts/summarization/run_headmask.py \
            --cad_alpha=0.5 --clamp_logit_diff=0 --mask_topk=50 \
            --model_nm llama2-chat --dataset_nm xsum \
            --decoding_algo cad_head_mask \
            --generation_config configs/topp_config.json \
            --output_dir /home/ameya/grounded-decoding/outputs/viz/xsum/ \
            --cad_kv_sharing total \
            --head_score_file configs/retrieval_head_score/llama-2-7b-80k.json \
            --return_js_divergence --output_scores --output_logits --output_attentions --output_hidden_states \
            --fp16 --n_samples 30 --shuffle_seed 1306 --topp 0.9 --temperature 1.0 --max_new_tokens 384
        ;;
esac