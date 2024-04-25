import accelerate
import argparse
import json
import logging
import os
import random
import torch
from tqdm import trange
from transformers import AutoTokenizer
from evaluate import load
import wandb
from src.contrastive_llama import LlamaTokenTypeAttnForCausalLM


logging.basicConfig(
    format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)
logger.debug("Imports done")


def main(args):
    if args.log_wandb:
        run = wandb.init(project="grounded-decoding", tags=["nq-swap", args.decoding_algo])
        sweep_or_run_id = run.sweep_id or run.id
        if sweep_or_run_id is not None:
            args.output_dir = os.path.join(args.output_dir, sweep_or_run_id)
        wandb.config.update(args, allow_val_change=True)

    logger.info(json.dumps(vars(args), indent=2))

    with open(args.generation_config) as fin:
        gen_config = json.load(fin)
    for k in gen_config:
        if k in args and getattr(args, k) is not None:
            gen_config[k] = getattr(args, k)
    logger.info(json.dumps(gen_config, indent=2))

    if args.log_wandb:
        wandb.config.update(gen_config, allow_val_change=True)

    device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

    with open(args.data_file) as fin:
        all_queries = [json.loads(line) for line in fin]
    if len(all_queries[0]) == 1 and 'dataset' in all_queries[0]:
        all_queries = all_queries[1:]
    if args.shuffle_seed is not None:
        random.Random(args.shuffle_seed).shuffle(all_queries)

    PROMPT_FORMAT = ["""[INST] <<SYS>> Give the answer to the user question based on the provided HTML passage. Output just the answer span.<</SYS>>""",
    """{}""",
    """Question: {} [/INST] Answer:"""]

    num_gpus = torch.cuda.device_count()
    accl_args = {"attn_implementation": "sdpa",
                 "device_map": "auto",
                 "offload_folder": f"/tmp/ameya/2024/offload/llama-2-7b-chat-hf/",
                 "max_memory": {i: f"48GiB" for i in range(num_gpus)},
                 }
    if args.fp16:
        accl_args["torch_dtype"] = torch.float16
    if args.log_wandb:
        wandb.config.update(accl_args)

    model = LlamaTokenTypeAttnForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", **accl_args)

    def tokenize_one_query(one_query, tokenizer):
        tokenized_parts = tokenizer([PROMPT_FORMAT[0], '\n'+one_query["context"], '\n' + PROMPT_FORMAT[2].format(one_query['query'])])
        full_input_ids = torch.LongTensor(tokenized_parts['input_ids'][0] + tokenized_parts['input_ids'][1][2:] + tokenized_parts['input_ids'][2][2:])[None, :]
        full_attn_mask = torch.LongTensor(tokenized_parts['attention_mask'][0] + tokenized_parts['attention_mask'][1][2:] + tokenized_parts['attention_mask'][2][2:])[None, :]
        context_mask = torch.LongTensor([0] * (len(tokenized_parts['input_ids'][0]) + 1) + [1] * (len(tokenized_parts['input_ids'][1][2:]) - 1) + [0] * len(tokenized_parts['input_ids'][2][2:]))[None, :]
        return {'input_ids': full_input_ids, 'attention_mask': full_attn_mask, 'context_attention_mask': context_mask}

    tokenized_prompts = [tokenize_one_query(all_queries[ex_id], tokenizer) for ex_id in trange(args.n_samples)]

    all_predictions = []
    predictions, references = [], []
    skipped_queries = 0
    for ex_id in trange(args.n_samples):
        one_tokenized_query_prompt = tokenized_prompts[ex_id]
        if one_tokenized_query_prompt['input_ids'].shape[1] > 4090:
            logger.debug(f"Skipping query: {ex_id} ({all_queries[ex_id]['uid']})")
            skipped_queries += 1
            continue
        if args.decoding_algo == "regular":
            output = model.generate(one_tokenized_query_prompt['input_ids'].to(device),
                                    **gen_config,
                                    output_scores=True, output_logits=True)
            output_logits = {"logits": [tok_logits.to('cpu').tolist() for tok_logits in output.logits]}
        elif args.decoding_algo == "self_cad_all_token":
            output = model.generate(one_tokenized_query_prompt['input_ids'].to(device),
                                    **gen_config, output_scores=True, output_logits=True,
                                    context_attention_mask=one_tokenized_query_prompt['context_attention_mask'].to(device),
                                    context_attention_weight=args.cad_beta, cad_alpha=args.cad_alpha)
            output_logits = {"logits": [tok_logits.to('cpu').tolist() for tok_logits in output.logits],
                             "base_logits": [tok_logits.to('cpu').tolist() for tok_logits in output.base_logits],
                             "perturbed_logits": [tok_logits.to('cpu').tolist() for tok_logits in output.perturbed_logits]}
        else:
            raise ValueError(f"Unsupported decoding algorithm: {args.decoding_algo}")
        
        one_prediction = {**all_queries[ex_id],
                          'model_output': tokenizer.batch_decode(output.sequences)[0],
                          'model_prediction': tokenizer.batch_decode(output.sequences[:, one_tokenized_query_prompt['input_ids'].shape[1]:])[0],
                          'response_logits': output_logits,
                          }
        all_predictions.append(one_prediction)
        prediction_text = one_prediction['model_prediction']
        prediction_text = prediction_text.replace('<span>', '').replace('</span>', '').replace('</s>', '')
        references.append({'id': all_queries[ex_id]['uid'],
                           'answers': [{"text": all_queries[ex_id]['gold_answers'][0]["text"], "answer_start": 0}]})
        predictions.append({'id': all_queries[ex_id]['uid'],
                            'prediction_text': prediction_text, 'no_answer_probability': 0.})

    if not os.path.isdir(args.output_dir):
        logger.info(f"Creating output dir: {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)
    outfile_nm = os.path.join(args.output_dir, f"{args.model_nm}-{args.decoding_algo}{'-topk' + str(args.topk) if args.topk is not None else ''}{'-topp' + str(args.topp) if args.topp is not None else ''}{'-temp' + str(args.temperature) if args.temperature is not None else ''}{'-alpha' + str(args.cad_alpha) if args.cad_alpha is not None else ''}{'-beta' + str(args.cad_beta) if args.cad_beta is not None else ''}-n{args.n_samples or ''}-r{args.shuffle_seed or ''}.jsonl")
    logger.info(f"Writing output to {outfile_nm}")
    with open(outfile_nm, 'w') as fout:
        out_str = '\n'.join([json.dumps(one_pred) for one_pred in all_predictions])
        fout.write(out_str)
    logger.info(f"Skipped queries: {skipped_queries}")

    squad_metric = load("squad_v2")
    pred_metrics = squad_metric.compute(predictions=predictions, references=references)
    logger.info(f"Evaluation results: {pred_metrics}")
    if args.log_wandb:
        wandb.log(pred_metrics)


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_nm", type=str, default="llama")
    parser.add_argument("--decoding_algo", type=str, choices=["regular", "self_cad_all_token"], required=True)
    parser.add_argument("--generation_config", type=str, required=True)
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--shuffle_seed", type=int, default=None)
    parser.add_argument("--fp16", action='store_true')
    parser.add_argument("--log_wandb", action='store_true')
    
    # CAD PARAMS
    parser.add_argument("--cad_alpha", type=float, default=None)
    parser.add_argument("--cad_beta", type=float, default=None)

    # OPTIONAL: Sampling arguments
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--topp", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)

    args_ = parser.parse_args()
    main(args_)
