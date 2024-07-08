import accelerate
import argparse
from functools import partial
import json
import logging
import numpy as np
import os
import random
import re
import string
import torch
from tqdm import trange
from transformers import AutoTokenizer
from datasets import Dataset, load_dataset
import evaluate
import wandb
from src.head_mask_llama import LlamaHeadMaskForCausalLM
from src.eval_utils import compute_factkb_score, compute_minicheck_score
from src.data_utils import preprocess_wikibio

logging.basicConfig(
    format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)
logger.debug("Imports done")


MODEL_MAP = {"llama2": "meta-llama/Llama-2-7b-hf",
             "llama2-chat": "meta-llama/Llama-2-7b-chat-hf"}
DATASET_HEADERS = {"cnn_dailymail": {"source": "article", "summary": "highlights"},
                   "xsum": {"source": "document", "summary": "summary"},
                   "wiki_bios": {"source": "user_message", "summary": None},
                   "ragtruth_news": {"source": "source_info", "summary": None},
                   "mediasum-tofu": {"source": "source", "query": "topic", "summary": None},
                   "meetingbank-tofu": {"source": "source", "query": "topic", "summary": None},
                   }
PROMPT_MAP = {("llama2", "cnn_dailymail"): ["""Article: {}""", """\nSummarize the article. Summary:"""],
              ("llama2", "xsum"): ["""Article: {}""", """Summarize the article in one sentence. Summary:"""],
              ("llama2-chat", "cnn_dailymail"): ["""[INST]<<SYS>>\nSummarize the article.\n<</SYS>>\n""", """{}""", """[/INST]Summary:"""],
            #   ("llama2-chat", "cnn_dailymail"): ["""Summarize the following news within 116 words:""", """{}""", """\noutput:"""],
              ("llama2-chat", "xsum"): ["""[INST]<<SYS>>\nSummarize the article in one sentence.\n<</SYS>>\n""", """{}""", """[/INST]Summary:"""],
              ("llama2-chat", "wiki_bios"): ["""[INST]<<SYS>>\nIn about 120 words, answer the user query based on the web snippets.\n<</SYS>>\n""", """{}""", """[/INST] Bio:"""],
              ("llama2-chat", "ragtruth_news"): ["""[INST]<<SYS>>\nSummarize the article.\n<</SYS>>\n""", """{}""", """[/INST]Summary:"""],
              ("llama2-chat", "mediasum-tofu-base"): ["""Document: {}""", """Summarize the provided document focusing on "{}". The summary should be less than 50 words in length.\nSummary:"""],
              ("llama2-chat", "mediasum-tofu-baseit"): ["""[INST]""", """Document:{}""", """Summarize the provided document focusing on "{}". The summary should be less than 50 words in length. [/INST]Summary:"""],
              ("llama2-chat", "meetingbank-tofu-base"): ["""Document: {}""", """Summarize the provided document focusing on "{}". The summary should be less than 50 words in length.\nSummary:"""],
              ("llama2-chat", "meetingbank-tofu-baseit"): ["""[INST]""", """Document: {}""", """Summarize the provided document focusing on "{}". The summary should be less than 50 words in length. [/INST]Summary:"""],}

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
    if args.dataset_nm == 'wiki_bios':
        logger.warn("Over-riding max_length to 4096.")
        gen_config['max_length'] = 4096
    logger.info(json.dumps(gen_config, indent=2))

    if args.log_wandb:
        wandb.config.update(gen_config, allow_val_change=True)

    # device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_MAP[args.model_nm])

    if args.dataset_nm == "cnn_dailymail":
        dataset = load_dataset("cnn_dailymail", "3.0.0", split='validation')
    elif args.dataset_nm == "xsum":
        dataset = load_dataset("EdinburghNLP/xsum", split='validation')
    elif args.dataset_nm == "wiki_bios":
        dataset = load_dataset("json",
                               data_files={
                                   "validation": "data/FActScore/search_results/labeled_you_search.jsonl",
                                   "test": "data/FActScore/search_results/unlabeled_you_search.jsonl"},
                               split="validation")
        dataset = dataset.map(partial(preprocess_wikibio, hits_per_query=10, snippets_per_hit=1),
                              load_from_cache_file=False)
        dataset = dataset.filter(lambda example: not example['to_drop'])
        logger.info(f"len(filtered dataset): {len(dataset)}")
    elif args.dataset_nm == "ragtruth_news":
        dset_raw = []
        with open("../RAGTruth/dataset/source_info.jsonl") as fin:
            for line in fin:
                one_ex = json.loads(line)
                if one_ex['source'] == "Recent News":
                    dset_raw.append(one_ex)
        dataset = Dataset.from_list(dset_raw)
        logger.info(f"len(dataset): {len(dataset)}")
    elif args.dataset_nm == "mediasum-tofu":
        dataset = load_dataset("json",
                               data_files={
                                   "validation": "data/tofueval/mediasum_dev_doc.jsonl",
                                   "test": "data/tofueval/mediasum_test_doc.jsonl"},
                               split="validation")
        logger.info(f"len(filtered dataset): {len(dataset)}")
    elif args.dataset_nm == "meetingbank-tofu":
        dataset = load_dataset("json",
                               data_files={
                                   "validation": "data/tofueval/meetingbank_dev_doc.jsonl",
                                   "test": "data/tofueval/meetingbank_test_doc.jsonl"},
                               split="validation")
        logger.info(f"len(filtered dataset): {len(dataset)}")
    else:
        raise ValueError(f"Unknown dataset {args.dataset_nm}")
    if args.shuffle_seed is not None:
        dataset = dataset.shuffle(seed=args.shuffle_seed)
    if args.n_samples is None:
        args.n_samples = len(dataset)

    num_gpus = torch.cuda.device_count()
    if args.decoding_algo == "cad_head_context_mask":
        attn_impl = "eager"
    elif args.fp16 and args.decoding_algo != "cad_head_context_mask":
        attn_impl = "flash_attention_2"
    else:
        attn_impl = "sdpa"
    accl_args = {"attn_implementation": attn_impl,
                 "device_map": "auto",
                 "offload_folder": f"/tmp/ameya/2024/offload/{MODEL_MAP[args.model_nm]}-{args.dataset_nm}{''.join(random.choices(string.ascii_uppercase + string.digits, k=5))}/",
                 "max_memory": {i: f"46GiB" for i in range(num_gpus)},
                 }
    if args.fp16:
        accl_args["torch_dtype"] = torch.float16
    if args.log_wandb:
        wandb.config.update(accl_args)

    model = LlamaHeadMaskForCausalLM.from_pretrained(MODEL_MAP[args.model_nm], **accl_args)
    model.eval()
    if args.decoding_algo in ("cad_head_mask", "cad_head_context_mask"):
        with open(args.head_score_file, "r") as file:
            stable_block_list =  json.loads(file.readline())
        stable_block_list = [(l[0], np.mean(l[1])) for l in stable_block_list.items()]
        stable_block_list = sorted(stable_block_list, key=lambda x: x[1], reverse=True)
        block_list = [[int(ll) for ll in l[0].split("-")] for l in stable_block_list][:args.mask_topk]
    else:
        block_list = None

    def tokenize_one_query(one_query, tokenizer, model_nm, dataset_nm, prompt_id=None):
        # NOTE: '\n' gets tokenized as ['_', '\n'] (i.e. 2 tokens) if it is at the start of the sentence
        # It will get tokenized as one token if there is a word before it without a space.
        # Ṣo, in the function below, we sometimes use [2:] for slicing to remove the space added by the Llama tokenizer
        if args.dataset_nm not in ("mediasum-tofu", "meetingbank-tofu",):
            PROMPT_FORMAT = PROMPT_MAP[(model_nm, dataset_nm)]
        else:
            PROMPT_FORMAT = PROMPT_MAP[(model_nm, f"{dataset_nm}-{prompt_id or 'base'}")]

        if model_nm == 'llama2':
            tokenized_parts = tokenizer([PROMPT_FORMAT[0].format(one_query[DATASET_HEADERS[dataset_nm]["source"]]),
                                         PROMPT_FORMAT[1]])
            full_input_ids = torch.LongTensor(tokenized_parts['input_ids'][0] + tokenized_parts['input_ids'][1][1:])[None, :]
            full_attn_mask = torch.LongTensor(tokenized_parts['attention_mask'][0] + tokenized_parts['attention_mask'][1][1:])[None, :]
            context_mask = torch.LongTensor([0] + [1] * (len(tokenized_parts['input_ids'][0]) - 1) + [0] * (len(tokenized_parts['input_ids'][1][1:])))[None, :]
        elif model_nm == 'llama2-chat':
            if args.dataset_nm not in ("mediasum-tofu", "meetingbank-tofu",):
                tokenized_parts = tokenizer([PROMPT_FORMAT[0],
                                            '\n'+one_query[DATASET_HEADERS[dataset_nm]["source"]],
                                            '\n'+PROMPT_FORMAT[2]])
                full_input_ids = torch.LongTensor(tokenized_parts['input_ids'][0] + tokenized_parts['input_ids'][1][2:] + tokenized_parts['input_ids'][2][2:])[None, :]
                full_attn_mask = torch.LongTensor(tokenized_parts['attention_mask'][0] + tokenized_parts['attention_mask'][1][2:] + tokenized_parts['attention_mask'][2][2:])[None, :]
                context_mask = torch.LongTensor([0] * (len(tokenized_parts['input_ids'][0]) + 1) + [1] * (len(tokenized_parts['input_ids'][1][2:]) - 1) + [0] * len(tokenized_parts['input_ids'][2][2:]))[None, :]
            else:
                if prompt_id is None or prompt_id == 'base':
                    tokenized_parts = tokenizer([PROMPT_FORMAT[0].format(one_query[DATASET_HEADERS[dataset_nm]["source"]]),
                                                '\n'+PROMPT_FORMAT[1].format(one_query[DATASET_HEADERS[dataset_nm]["topic"]])])
                    full_input_ids = torch.LongTensor(tokenized_parts['input_ids'][0] + tokenized_parts['input_ids'][1][2:])[None, :]
                    full_attn_mask = torch.LongTensor(tokenized_parts['attention_mask'][0] + tokenized_parts['attention_mask'][1][2:])[None, :]
                    context_mask = torch.LongTensor([0] + [1] * (len(tokenized_parts['input_ids'][0][1:])) + [0] * (len(tokenized_parts['input_ids'][1][2:])))[None, :]
                elif prompt_id == 'base-it':
                    tokenized_parts = tokenizer([PROMPT_FORMAT[0],
                                                '\n'+PROMPT_FORMAT[1].format(one_query[DATASET_HEADERS[dataset_nm]["source"]]),
                                                '\n'+PROMPT_FORMAT[2].format(one_query[DATASET_HEADERS[dataset_nm]["topic"]])])
                    full_input_ids = torch.LongTensor(tokenized_parts['input_ids'][0] + tokenized_parts['input_ids'][1][2:] + tokenized_parts['input_ids'][2][2:])[None, :]
                    full_attn_mask = torch.LongTensor(tokenized_parts['attention_mask'][0] + tokenized_parts['attention_mask'][1][2:] + tokenized_parts['attention_mask'][2][2:])[None, :]
                    context_mask = torch.LongTensor([0] * (len(tokenized_parts['input_ids'][0]) + 1) + [1] * (len(tokenized_parts['input_ids'][1][2:]) - 1) + [0] * len(tokenized_parts['input_ids'][2][2:]))[None, :]
                else:
                    raise ValueError(f"Unknown prompt id: {prompt_id}")
        else:
            raise ValueError(f"Unknown model name: {model_nm}")
        assert full_input_ids.shape == context_mask.shape == full_attn_mask.shape
        return {'input_ids': full_input_ids, 'attention_mask': full_attn_mask, 'context_attention_mask': context_mask}

    tokenized_prompts = [tokenize_one_query(dataset[ex_id], tokenizer, model_nm=args.model_nm, dataset_nm=args.dataset_nm, prompt_id=args.prompt_id)
                         for ex_id in trange(args.n_samples)]
    if args.return_js_divergence:
        js_divergences_arr = []
        js_divergences_output_arr = []

    all_predictions = []
    predictions, references, articles = [], [], []
    skipped_queries = 0
    with torch.no_grad():
        for ex_id in trange(args.n_samples):
            one_tokenized_query_prompt = tokenized_prompts[ex_id]
            if ex_id == 0:
                logger.info(tokenizer.batch_decode(one_tokenized_query_prompt["input_ids"])[0])
            if one_tokenized_query_prompt['input_ids'].shape[1] > (gen_config['max_length'] - gen_config['max_new_tokens']):
                logger.debug(f"Skipping query: {ex_id} ({dataset[ex_id]['id']})")
                skipped_queries += 1
                continue
            if args.decoding_algo == "regular":
                output = model.generate(one_tokenized_query_prompt['input_ids'].to('cuda'),
                                        **gen_config,
                                        output_scores=args.output_scores, output_logits=args.output_logits,
                                        output_hidden_states=args.output_hidden_states, output_attentions=args.output_attentions)
                if args.output_scores:
                    output_scores = {"scores": [tok_scores.to('cpu').tolist() for tok_scores in output.scores]}
                if args.output_logits:
                    output_logits = {"logits": [tok_logits.to('cpu').tolist() for tok_logits in output.logits]}
                if args.return_js_divergence:
                    js_divergences = None
                    js_divergences_output = None
            elif args.decoding_algo in ("cad_head_mask", "cad_head_context_mask"):
                if args.decoding_algo == "cad_head_mask":
                    masking_kwargs = {"block_list": block_list}
                elif args.decoding_algo == "cad_head_context_mask":
                    masking_kwargs = {"block_list": block_list,
                                      "context_attention_mask": one_tokenized_query_prompt['context_attention_mask'].to('cuda'),
                                      "context_attention_weight": args.cad_beta}
                output = model.generate(one_tokenized_query_prompt['input_ids'].to('cuda'),
                                        **gen_config, output_scores=args.output_scores, output_logits=args.output_logits,
                                        output_hidden_states=args.output_hidden_states, output_attentions=args.output_attentions,
                                        cad_alpha=args.cad_alpha, cad_kv_sharing=args.cad_kv_sharing,
                                        return_js_divergence=args.return_js_divergence,
                                        clamp_logit_diff=args.clamp_logit_diff,
                                        **masking_kwargs)
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
            
            one_prediction = {**dataset[ex_id],
                              'model_output': tokenizer.batch_decode(output.sequences)[0],
                              'model_prediction': tokenizer.batch_decode(output.sequences[:, one_tokenized_query_prompt['input_ids'].shape[1]:])[0],
                              'response_scores': output_scores if args.output_scores else None,
                              'response_logits': output_logits if args.output_logits else None,
                              'js_divergences': js_divergences if args.return_js_divergence else None,
                              'js_divergences_output': js_divergences_output if args.return_js_divergence else None,
                             }
            if args.dataset_nm == 'wiki_bios':
                one_prediction['output'] = one_prediction['model_prediction'].replace('<span>', '').replace('</span>', '').replace('</s>', '').strip()
                one_prediction['topic'] = re.fullmatch(r"Tell me a bio of (.*).", one_prediction['query']).group(1)
            all_predictions.append(one_prediction)
            prediction_text = one_prediction['model_prediction']
            prediction_text = prediction_text.replace('<span>', '').replace('</span>', '').replace('</s>', '')
            if DATASET_HEADERS[args.dataset_nm]["summary"]:
                references.append(dataset[ex_id][DATASET_HEADERS[args.dataset_nm]["summary"]])
            predictions.append(prediction_text)
            articles.append(dataset[ex_id][DATASET_HEADERS[args.dataset_nm]["source"]])

    if not os.path.isdir(args.output_dir):
        logger.info(f"Creating output dir: {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)
    outfile_nm = os.path.join(args.output_dir, f"{args.model_nm}-{args.decoding_algo}{'-topk' + str(args.topk) if args.topk is not None else ''}{'-topp' + str(args.topp) if args.topp is not None else ''}{'-temp' + str(args.temperature) if args.temperature is not None else ''}{'-alpha' + str(args.cad_alpha) if args.cad_alpha is not None else ''}{'-mask_topk' + str(args.mask_topk) if args.mask_topk is not None else ''}-n{args.n_samples or ''}-r{args.shuffle_seed or ''}.jsonl")
    logger.info(f"Writing output to {outfile_nm}")
    with open(outfile_nm, 'w') as fout:
        out_str = '\n'.join([json.dumps(one_pred) for one_pred in all_predictions])
        fout.write(out_str)
    logger.info(f"Skipped queries: {skipped_queries}")

    pred_metric = {}

    bert_metric = evaluate.load("bertscore")
    bert_res = bert_metric.compute(predictions=predictions, references=articles, model_type="microsoft/deberta-xlarge-mnli")
    pred_metric["bertscore"] = {k: np.mean(bert_res[k]) for k in ["precision", "recall", "f1"]}

    pred_metric["factkb"] = compute_factkb_score(predictions=predictions, articles=articles)

    minicheck_metric = {}
    for minicheck_modelnm in ['roberta-large', 'deberta-v3-large', 'flan-t5-large']:
        # output[1] is log which is ignored for now
        _mini_ais, _ = compute_minicheck_score(predictions=predictions, articles=articles,
                                               scorer_nm=minicheck_modelnm, use_cuda=(num_gpus > 0))
        minicheck_metric[minicheck_modelnm] = _mini_ais
    pred_metric['minicheck'] = minicheck_metric

    if len(references) > 0:
        rouge = evaluate.load('rouge')
        rouge_res = rouge.compute(predictions=predictions, references=references,
                                rouge_types=['rouge1', 'rouge2', 'rougeL'], use_aggregator=True)
        pred_metric["rouge"] = rouge_res

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
    parser.add_argument("--dataset_nm", type=str, choices=["cnn_dailymail", "xsum", "wiki_bios", "ragtruth_news", "mediasum-tofu", "meetingbank-tofu"], required=True)
    parser.add_argument("--prompt_id", type=str, required=False, help="Specific prompt identifier for meetingbank and mediasum.")
    parser.add_argument("--decoding_algo", type=str, choices=["regular", "cad_head_mask", "cad_head_context_mask"], required=True)
    parser.add_argument("--generation_config", type=str, required=True)
    
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--shuffle_seed", type=int, default=None)
    parser.add_argument("--fp16", action='store_true')
    parser.add_argument("--log_wandb", action='store_true')
    
    # CAD PARAMS
    parser.add_argument("--cad_alpha", type=float, default=None)
    parser.add_argument("--cad_beta", type=float, default=None,
                        help="Only used with `cad_head_context_mask`")
    parser.add_argument("--cad_kv_sharing", type=str, default="total")
    parser.add_argument("--head_score_file", type=str, default="")
    parser.add_argument("--mask_topk", type=int, default=None)
    parser.add_argument("--clamp_logit_diff", type=int, choices=[0, 1],
                        help='Clamp (base-perturbed) logits to min value 0. Currently only works for `cad` modes.')

    # Analysis options
    parser.add_argument("--return_js_divergence", action='store_true')
    parser.add_argument("--output_hidden_states", action='store_true')
    parser.add_argument("--output_attentions", action='store_true')
    parser.add_argument("--output_scores", action='store_true')
    parser.add_argument("--output_logits", action='store_true')

    # OPTIONAL: Sampling arguments
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--topp", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)

    args_ = parser.parse_args()
    
    args_.clamp_logit_diff = bool(args_.clamp_logit_diff)
    main(args_)
