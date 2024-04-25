import accelerate
import argparse
import json
import logging
import numpy as np
import os
import random
import torch
from tqdm import trange
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import evaluate
import wandb
from src.contrastive_llama import LlamaTokenTypeAttnForCausalLM


logging.basicConfig(
    format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)
logger.debug("Imports done")


MODEL_MAP = {"llama2": "meta-llama/Llama-2-7b-hf",
             "llama2-chat": "meta-llama/Llama-2-7b-chat-hf"}
DATASET_HEADERS = {"cnn_dailymail": {"source": "article", "summary": "highlights"},
                   "xsum": {"source": "document", "summary": "summary"}}
PROMPT_MAP = {("llama2", "cnn_dailymail"): ["""Article: {}""", """\nSummarize the article. Summary:"""],
              ("llama2", "xsum"): ["""Article: {}""", """Summarize the article in one sentence. Summary:"""],
              ("llama2-chat", "cnn_dailymail"): ["""[INST] <<SYS>> Summarize the article.<</SYS>>""", """{}""", """[/INST] Summary:"""],
              ("llama2-chat", "xsum"): ["""[INST] <<SYS>> Summarize the article in one sentence.<</SYS>>""", """{}""", """[/INST] Summary:"""]}


def compute_factkb(predictions, articles, model_kwargs=None, batch_size=16):
    inputs = [[pred, context] for pred, context in zip(predictions, articles)]

    tokenizer = AutoTokenizer.from_pretrained("roberta-base", padding="max_length", truncation=True)
    factkb = AutoModelForSequenceClassification.from_pretrained("bunsenfeng/FactKB", num_labels=2)

    results = []
    with torch.no_grad():
        for bctr in range(int(np.ceil(len(inputs)/batch_size))):
            batch_inputs = inputs[bctr*batch_size:(bctr+1)*batch_size]
            tokens = tokenizer(batch_inputs, return_tensors="pt", padding="max_length", truncation=True)
            result = torch.softmax(factkb(**tokens).logits, dim = 1)
            results.extend(result[:,1].cpu().tolist())
    
    return {'factkb': np.mean(results)}


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

    if args.dataset_nm == "cnn_dailymail":
        dataset = load_dataset("cnn_dailymail", "3.0.0", split='validation')
    elif args.dataset_nm == "xsum":
        dataset = load_dataset("EdinburghNLP/xsum", split='validation')
    else:
        raise ValueError(f"Unknown dataset {args.dataset_nm}")
    if args.shuffle_seed is not None:
        dataset = dataset.shuffle(seed=args.shuffle_seed)
    if args.n_samples is None:
        args.n_samples = len(dataset)

    PROMPT_FORMAT = PROMPT_MAP[(args.model_nm, args.dataset_nm)]

    num_gpus = torch.cuda.device_count()
    accl_args = {"attn_implementation": "sdpa",
                 "device_map": "auto",
                 "offload_folder": f"/tmp/ameya/2024/offload/{MODEL_MAP[args.model_nm]}-{args.dataset_nm}{random.randint(0,5)}/",
                 "max_memory": {i: f"48GiB" for i in range(num_gpus)},
                 }
    if args.fp16:
        accl_args["torch_dtype"] = torch.float16
    if args.log_wandb:
        wandb.config.update(accl_args)

    model = LlamaTokenTypeAttnForCausalLM.from_pretrained(MODEL_MAP[args.model_nm], **accl_args)

    def tokenize_one_query(one_query, tokenizer, model_nm, dataset_nm):
        if model_nm == 'llama2':
            tokenized_parts = tokenizer([PROMPT_FORMAT[0].format(one_query[DATASET_HEADERS[dataset_nm]["source"]]),
                                         PROMPT_FORMAT[1]])
            full_input_ids = torch.LongTensor(tokenized_parts['input_ids'][0] + tokenized_parts['input_ids'][1][1:])[None, :]
            full_attn_mask = torch.LongTensor(tokenized_parts['attention_mask'][0] + tokenized_parts['attention_mask'][1][1:])[None, :]
            context_mask = torch.LongTensor([0] + [1] * (len(tokenized_parts['input_ids'][0]) - 1) + [0] * (len(tokenized_parts['input_ids'][1][1:])))[None, :]
        elif model_nm == 'llama2-chat':
            tokenized_parts = tokenizer([PROMPT_FORMAT[0],
                                         '\n'+one_query[DATASET_HEADERS[dataset_nm]["source"]],
                                         '\n'+PROMPT_FORMAT[2]])
            full_input_ids = torch.LongTensor(tokenized_parts['input_ids'][0] + tokenized_parts['input_ids'][1][2:] + tokenized_parts['input_ids'][2][2:])[None, :]
            full_attn_mask = torch.LongTensor(tokenized_parts['attention_mask'][0] + tokenized_parts['attention_mask'][1][2:] + tokenized_parts['attention_mask'][2][2:])[None, :]
            context_mask = torch.LongTensor([0] * (len(tokenized_parts['input_ids'][0]) + 1) + [1] * (len(tokenized_parts['input_ids'][1][2:]) - 1) + [0] * len(tokenized_parts['input_ids'][2][2:]))[None, :]
        else:
            raise ValueError(f"Unknown model name: {model_nm}")
        assert full_input_ids.shape == context_mask.shape == full_attn_mask.shape
        return {'input_ids': full_input_ids, 'attention_mask': full_attn_mask, 'context_attention_mask': context_mask}

    tokenized_prompts = [tokenize_one_query(dataset[ex_id], tokenizer, model_nm=args.model_nm, dataset_nm=args.dataset_nm)
                          for ex_id in trange(args.n_samples)]

    all_predictions = []
    predictions, references, articles = [], [], []
    skipped_queries = 0
    for ex_id in trange(args.n_samples):
        one_tokenized_query_prompt = tokenized_prompts[ex_id]
        if ex_id == 0:
            logger.info(tokenizer.batch_decode(one_tokenized_query_prompt["input_ids"])[0])
        if one_tokenized_query_prompt['input_ids'].shape[1] > 4090:
            logger.debug(f"Skipping query: {ex_id} ({dataset[ex_id]['id']})")
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
        elif args.decoding_algo == "self_cad_all_token":
            output = model.generate(one_tokenized_query_prompt['input_ids'].to(device),
                                    **gen_config, output_scores=args.output_scores, output_logits=args.output_logits,
                                    context_attention_mask=one_tokenized_query_prompt['context_attention_mask'].to(device),
                                    context_attention_weight=args.cad_beta, cad_alpha=args.cad_alpha)
            if args.output_scores:
                output_scores = {"scores": [tok_scores.to('cpu').tolist() for tok_scores in output.scores]}
            if args.output_logits:
                output_logits = {"logits": [tok_logits.to('cpu').tolist() for tok_logits in output.logits],
                                "base_logits": [tok_logits.to('cpu').tolist() for tok_logits in output.base_logits],
                                "perturbed_logits": [tok_logits.to('cpu').tolist() for tok_logits in output.perturbed_logits]}
        else:
            raise ValueError(f"Unsupported decoding algorithm: {args.decoding_algo}")
        
        one_prediction = {**dataset[ex_id],
                          'model_output': tokenizer.batch_decode(output.sequences)[0],
                          'model_prediction': tokenizer.batch_decode(output.sequences[:, one_tokenized_query_prompt['input_ids'].shape[1]:])[0],
                          'response_scores': output_scores if args.output_scores else None,
                          'response_logits': output_logits if args.output_logits else None,
                          }
        all_predictions.append(one_prediction)
        prediction_text = one_prediction['model_prediction']
        prediction_text = prediction_text.replace('<span>', '').replace('</span>', '').replace('</s>', '')
        references.append(dataset[ex_id][DATASET_HEADERS[args.dataset_nm]["summary"]])
        predictions.append(prediction_text)
        articles.append(dataset[ex_id][DATASET_HEADERS[args.dataset_nm]["source"]])

    if not os.path.isdir(args.output_dir):
        logger.info(f"Creating output dir: {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)
    outfile_nm = os.path.join(args.output_dir, f"{args.model_nm}-{args.decoding_algo}{'-topk' + str(args.topk) if args.topk is not None else ''}{'-topp' + str(args.topp) if args.topp is not None else ''}{'-temp' + str(args.temperature) if args.temperature is not None else ''}{'-alpha' + str(args.cad_alpha) if args.cad_alpha is not None else ''}{'-beta' + str(args.cad_beta) if args.cad_beta is not None else ''}-n{args.n_samples or ''}-r{args.shuffle_seed or ''}.jsonl")
    logger.info(f"Writing output to {outfile_nm}")
    with open(outfile_nm, 'w') as fout:
        out_str = '\n'.join([json.dumps(one_pred) for one_pred in all_predictions])
        fout.write(out_str)
    logger.info(f"Skipped queries: {skipped_queries}")

    bert_metric = evaluate.load("bertscore")
    pred_metric = {}
    bert_res = bert_metric.compute(predictions=predictions, references=references, model_type="microsoft/deberta-xlarge-mnli")
    pred_metric["bertscore"] = {k: np.mean(bert_res[k]) for k in ["precision", "recall", "f1"]}
    pred_metric["factkb"] = compute_factkb(predictions=predictions, articles=articles)
    rouge = evaluate.load('rouge')
    rouge_res = rouge.compute(predictions=predictions, references=references,
                              rouge_types=['rouge1', 'rouge2', 'rougeL'], use_aggregator=True)
    pred_metric["rouge"] = rouge_res
    logger.info(f"Evaluation results: {pred_metric}")
    if args.log_wandb:
        wandb.log(pred_metric)


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_nm", type=str, choices=["llama2", "llama2-chat"], default="llama2-chat")
    parser.add_argument("--dataset_nm", type=str, choices=["cnn_dailymail", "xsum"], required=True)
    parser.add_argument("--decoding_algo", type=str, choices=["regular", "self_cad_all_token"], required=True)
    parser.add_argument("--generation_config", type=str, required=True)
    
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_scores", action='store_true')
    parser.add_argument("--output_logits", action='store_true')

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
