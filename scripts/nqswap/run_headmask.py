import accelerate
import argparse
import json
import logging
import numpy as np
import os
import random
import string
import torch
from tqdm import trange
from transformers import AutoTokenizer
from evaluate import load
import wandb
from src.head_mask_llama import LlamaHeadMaskForCausalLM


logging.basicConfig(
    format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)
logger.debug("Imports done")


MODEL_MAP = {"llama2-chat": "meta-llama/Llama-2-7b-chat-hf"}
PROMPT_MAP = {("llama2-chat", "nq"): ["""[INST] <<SYS>> Give the answer to the user question based on the provided HTML passage. Output just the answer span.<</SYS>>""",
                                      """{}""",
                                      """Question: {} [/INST] Answer:"""],
                ("llama2-chat", "nq-swap"): ["""[INST] <<SYS>> Give the answer to the user question based on the provided HTML passage. Output just the answer span.<</SYS>>""",
                                      """{}""",
                                      """Question: {} [/INST] Answer:"""]}


def main(args):
    if args.log_wandb:
        run = wandb.init(project="grounded-decoding", tags=[args.dataset_nm, args.decoding_algo])
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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_MAP[args.model_nm])

    with open(args.data_file) as fin:
        all_queries = [json.loads(line) for line in fin]
    if len(all_queries[0]) == 1 and 'dataset' in all_queries[0]:
        all_queries = all_queries[1:]
    if args.shuffle_seed is not None:
        random.Random(args.shuffle_seed).shuffle(all_queries)

    PROMPT_FORMAT = PROMPT_MAP[(args.model_nm, args.dataset_nm)]

    num_gpus = torch.cuda.device_count()
    accl_args = {"attn_implementation": "sdpa",
                 "device_map": "auto",
                 "offload_folder": f"/tmp/ameya/2024/offload/{MODEL_MAP[args.model_nm]}-{args.dataset_nm}{''.join(random.choices(string.ascii_uppercase + string.digits, k=5))}/",
                 "max_memory": {i: f"48GiB" for i in range(num_gpus)},
                 }
    if args.fp16:
        accl_args["torch_dtype"] = torch.float16
    if args.log_wandb:
        wandb.config.update(accl_args)

    model = LlamaHeadMaskForCausalLM.from_pretrained(MODEL_MAP[args.model_nm], **accl_args)
    if args.decoding_algo == "cad_head_mask":
        with open(args.head_score_file, "r") as file:
            stable_block_list =  json.loads(file.readline())
        stable_block_list = [(l[0], np.mean(l[1])) for l in stable_block_list.items()]
        stable_block_list = sorted(stable_block_list, key=lambda x: x[1], reverse=True)
        block_list = [[int(ll) for ll in l[0].split("-")] for l in stable_block_list][:args.mask_topk]
    else:
        block_list = None

    def tokenize_one_query(one_query, tokenizer):
        tokenized_parts = tokenizer([PROMPT_FORMAT[0], '\n'+one_query["context"], '\n' + PROMPT_FORMAT[2].format(one_query['query'])])
        full_input_ids = torch.LongTensor(tokenized_parts['input_ids'][0] + tokenized_parts['input_ids'][1][2:] + tokenized_parts['input_ids'][2][2:])[None, :]
        full_attn_mask = torch.LongTensor(tokenized_parts['attention_mask'][0] + tokenized_parts['attention_mask'][1][2:] + tokenized_parts['attention_mask'][2][2:])[None, :]
        context_mask = torch.LongTensor([0] * (len(tokenized_parts['input_ids'][0]) + 1) + [1] * (len(tokenized_parts['input_ids'][1][2:]) - 1) + [0] * len(tokenized_parts['input_ids'][2][2:]))[None, :]
        return {'input_ids': full_input_ids, 'attention_mask': full_attn_mask, 'context_attention_mask': context_mask}

    tokenized_prompts = [tokenize_one_query(all_queries[ex_id], tokenizer) for ex_id in trange(args.n_samples)]
    if args.return_js_divergence:
        js_divergences_arr = []
        js_divergences_output_arr = []

    all_predictions = []
    predictions, references = [], []
    skipped_queries = 0
    for ex_id in trange(args.n_samples):
        one_tokenized_query_prompt = tokenized_prompts[ex_id]
        if ex_id == 0:
            logger.info(tokenizer.batch_decode(one_tokenized_query_prompt["input_ids"])[0])
        if one_tokenized_query_prompt['input_ids'].shape[1] > 4090:
            logger.debug(f"Skipping query: {ex_id} ({all_queries[ex_id]['uid']})")
            skipped_queries += 1
            continue
        if args.decoding_algo == "regular":
            output = model.generate(one_tokenized_query_prompt['input_ids'].to(device),
                                    **gen_config,
                                    output_scores=args.output_scores, output_logits=args.output_logits)
            if args.output_scores:
                output_scores = {"scores": [tok_scores.to('cpu').tolist() for tok_scores in output.scores]}
            if args.output_logits:
                output_logits = {"logits": [tok_logits.to('cpu').tolist() for tok_logits in output.logits]}
            if args.return_js_divergence:
                js_divergences = None
                js_divergences_output = None
        elif args.decoding_algo == "cad_head_mask":
            output = model.generate(one_tokenized_query_prompt['input_ids'].to(device),
                                    **gen_config, output_scores=args.output_scores, output_logits=args.output_logits,
                                    cad_alpha=args.cad_alpha, cad_kv_sharing=args.cad_kv_sharing,
                                    return_js_divergence=args.return_js_divergence, block_list=block_list)
            output_logits = {"logits": [tok_logits.to('cpu').tolist() for tok_logits in output.logits],
                             "base_logits": [tok_logits.to('cpu').tolist() for tok_logits in output.base_logits],
                             "perturbed_logits": [tok_logits.to('cpu').tolist() for tok_logits in output.perturbed_logits]}
            if args.output_scores:
                output_scores = {"scores": [tok_scores.to('cpu').tolist() for tok_scores in output.scores]}
            if args.output_logits:
                output_logits = {"logits": [tok_logits.to('cpu').tolist() for tok_logits in output.logits],
                                "base_logits": [tok_logits.to('cpu').tolist() for tok_logits in output.base_logits],
                                "perturbed_logits": [tok_logits.to('cpu').tolist() for tok_logits in output.perturbed_logits]}
            if args.return_js_divergence:
                js_divergences = output.payload['js_divergence']
                js_divergences_arr.append(np.array(js_divergences).reshape(-1))
                js_divergences_output = output.payload['js_divergence_output']
                js_divergences_output_arr.append(np.array(js_divergences_output).reshape(-1))
        else:
            raise ValueError(f"Unsupported decoding algorithm: {args.decoding_algo}")
        
        one_prediction = {**all_queries[ex_id],
                          'model_output': tokenizer.batch_decode(output.sequences)[0],
                          'model_prediction': tokenizer.batch_decode(output.sequences[:, one_tokenized_query_prompt['input_ids'].shape[1]:])[0],
                          'response_scores': output_scores if args.output_scores else None,
                          'response_logits': output_logits if args.output_logits else None,
                          'js_divergences': js_divergences if args.return_js_divergence else None,
                          'js_divergences_output': js_divergences_output if args.return_js_divergence else None,
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
    outfile_nm = os.path.join(args.output_dir, f"{args.model_nm}-{args.decoding_algo}{'-topk' + str(args.topk) if args.topk is not None else ''}{'-topp' + str(args.topp) if args.topp is not None else ''}{'-temp' + str(args.temperature) if args.temperature is not None else ''}{'-alpha' + str(args.cad_alpha) if args.cad_alpha is not None else ''}{'-mask_topk' + str(args.mask_topk) if args.mask_topk is not None else ''}-n{args.n_samples or ''}-r{args.shuffle_seed or ''}.jsonl")
    logger.info(f"Writing output to {outfile_nm}")
    with open(outfile_nm, 'w') as fout:
        out_str = '\n'.join([json.dumps(one_pred) for one_pred in all_predictions])
        fout.write(out_str)
    logger.info(f"Skipped queries: {skipped_queries}")

    squad_metric = load("squad_v2")
    pred_metric = squad_metric.compute(predictions=predictions, references=references)

    if args.return_js_divergence:
        avg_js_divergences_arr = [np.mean(one_js_div) for one_js_div in js_divergences_arr]
        q90_js_divergences_arr = [np.quantile(one_js_div, 0.9) for one_js_div in js_divergences_arr]
        pred_metric["mean_js_divergence"] = np.mean(avg_js_divergences_arr)
        pred_metric["q90_js_divergence"] = np.quantile(q90_js_divergences_arr, 0.9)

        avg_js_divergences_output_arr = [np.mean(one_js_div) for one_js_div in js_divergences_output_arr]
        q90_js_divergences_output_arr = [np.quantile(one_js_div, 0.9) for one_js_div in js_divergences_output_arr]
        pred_metric["mean_js_divergence_base_output"] = np.mean(avg_js_divergences_output_arr)
        pred_metric["q90_js_divergence_base_output"] = np.quantile(q90_js_divergences_output_arr, 0.9)

        # max_len = max([len(one_js_div) for one_js_div in js_divergences_arr])
        # padded_js_div_arr = np.zeros((len(js_divergences_arr), max_len))
        # for row_ctr, one_js_div in enumerate(js_divergences_arr):
        #     padded_js_div_arr[row_ctr, :len(one_js_div)] = one_js_div
        # pred_metric["js_divergences_table"] = wandb.Table(columns=np.arange(max_len).tolist(), data=padded_js_div_arr.tolist())
    
    logger.info(f"Evaluation results: {pred_metric}")
    if args.log_wandb:
        wandb.log(pred_metric)


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_nm", type=str, choices=["llama2", "llama2-chat"], default="llama2-chat")
    parser.add_argument("--dataset_nm", type=str, choices=["nq", "nq-swap"], required=True)
    parser.add_argument("--decoding_algo", type=str, choices=["regular", "cad_head_mask"], required=True)
    parser.add_argument("--generation_config", type=str, required=True)
    parser.add_argument("--data_file", type=str, required=True)

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_scores", action='store_true')
    parser.add_argument("--output_logits", action='store_true')

    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--shuffle_seed", type=int, default=None)
    parser.add_argument("--fp16", action='store_true')
    parser.add_argument("--log_wandb", action='store_true')
    
    # CAD PARAMS
    parser.add_argument("--cad_alpha", type=float, default=None)
    parser.add_argument("--cad_kv_sharing", type=str, default="total")
    parser.add_argument("--head_score_file", type=str, default="")
    parser.add_argument("--mask_topk", type=int, default=None)

    # Analysis options
    parser.add_argument("--return_js_divergence", action='store_true')

    # OPTIONAL: Sampling arguments
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--topp", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)

    args_ = parser.parse_args()
    main(args_)
