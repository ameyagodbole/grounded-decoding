import argparse
from datasets import load_dataset
import json
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import re
import string
from tqdm import trange
import os

from src.data_utils import preprocess_wikibio


PROMPT_MAP = {("huggyllama/llama-7b", "cnn_dailymail"): {'article_frag': 0, 'other_frags': [1], 'fragments': ["""Article: {}\n""", """Summarize the article. Summary:"""]},
              ("huggyllama/llama-7b", "xsum"): {'article_frag': 0, 'other_frags': [1], 'fragments': ["""Article: {}\n""", """Summarize the article in one sentence. Summary:"""]},
              ("meta-llama/Llama-2-7b-chat-hf", "cnn_dailymail"): {'article_frag': 1, 'other_frags': [0, 2], 'fragments': ["""[INST]<<SYS>>\nSummarize the article.\n<</SYS>>\n""", """{}\n""", """[/INST]Summary:"""]},
              ("meta-llama/Llama-2-7b-chat-hf", "xsum"): {'article_frag': 1, 'other_frags': [0, 2], 'fragments': ["""[INST]<<SYS>>\nSummarize the article in one sentence.\n<</SYS>>\n""", """{}\n""", """[/INST]Summary:"""]},
              ("meta-llama/Llama-2-7b-chat-hf", "wiki_bios"): {'article_frag': 1, 'other_frags': [0, 2], 'fragments': ["""[INST]<<SYS>>\nIn about 120 words, answer the user query based on the web snippets.\n<</SYS>>\n""", """{}""", """[/INST] Bio:"""]}
              }
DATASET_HEADERS = {"cnn_dailymail": {"source": "article", "summary": "highlights"},
                   "xsum": {"source": "document", "summary": "summary"},
                   "wiki_bios": {"source": "user_message", "summary": None}}


def shuffle_summ(context, shuffle_type):
    if shuffle_type == "shuffle_sents":
        context_sents = sent_tokenize(context)
        np.random.shuffle(context_sents)
        shuffled_sents = list(context_sents)
        return ' '.join(shuffled_sents)
    elif shuffle_type == "shuffle_sent_words":
        sent_words = [word_tokenize(one_sent) for one_sent in sent_tokenize(context)]
        shuffled_sents = []
        for one_sent_words in sent_words:
            np.random.shuffle(one_sent_words)
            shuffled_sents.append(' '.join(one_sent_words))
        return ' '.join(shuffled_sents)
    elif shuffle_type == "shuffle_punkt":
        sent_words = [word_tokenize(one_sent) for one_sent in sent_tokenize(context)]
        shuffled_sents = []
        for one_sent_words in sent_words:
            sent_phrases = []
            curr_phrase = ''
            for one_word in one_sent_words:
                if len(one_word) == 1 and one_word in string.punctuation + '–':
                    curr_phrase += one_word
                    sent_phrases.append(curr_phrase)
                    curr_phrase = ''
                else:
                    curr_phrase += f" {one_word}"
            if curr_phrase:
                sent_phrases.append(curr_phrase)
            np.random.shuffle(sent_phrases)
            shuffled_sents.append(' '.join(list(sent_phrases)))
        return ' '.join(shuffled_sents)
    else:
        raise NotImplementedError(f"Unknown type: {shuffle_type}")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_nm", type=str, choices=["cnn_dailymail", "xsum", "wiki_bios"], required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--model_nm", type=str, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--shuffle_seed", type=int, default=None)

    parser.add_argument("--perturb_input2", type=str, default=None,
                        choices=["shuffle_sents", "shuffle_sent_words", "shuffle_punkt"])
    args = parser.parse_args()

    print("Args:")
    print(args)

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
        dataset = dataset.map(preprocess_wikibio)  # Can pass arguments with `fn_kwargs`
        dataset = dataset.filter(lambda example: not example['to_drop'])
        print(f"len(filtered dataset): {len(dataset)}")
    else:
        raise ValueError(f"Unknown dataset {args.dataset_nm}")
    if args.shuffle_seed is not None:
        dataset = dataset.shuffle(seed=args.shuffle_seed)
    if args.n_samples is None:
        args.n_samples = len(dataset)

    output_dir = os.path.dirname(args.output_file)
    if not os.path.isdir(output_dir):
        print(f"Creating output dir: {output_dir}")
        os.makedirs(output_dir)
    
    prompt_format = PROMPT_MAP[(args.model_nm, args.dataset_nm)]
    frags_before_article = ''.join([prompt_format['fragments'][f_id] for f_id in prompt_format['other_frags'] if f_id < prompt_format['article_frag']])
    article_frag = prompt_format['fragments'][prompt_format['article_frag']]
    frags_after_article = ''.join([prompt_format['fragments'][f_id] for f_id in prompt_format['other_frags'] if f_id > prompt_format['article_frag']])

    out_data = []
    for ex_id in trange(args.n_samples):
        out_dict0 = {"input_index": ex_id,
                     "assigned_model": args.model_nm,
                     "assigned_process": 0,
                     "context_string": frags_before_article + article_frag.format(dataset[ex_id][DATASET_HEADERS[args.dataset_nm]['source']]) + frags_after_article,
                     "assigned_weight": 1.0 + args.alpha,
                     "filter_p": 0.9,
                     "orig_id": dataset[ex_id]['id'],
                     "article": dataset[ex_id][DATASET_HEADERS[args.dataset_nm]['source']],
                     "gold_answers": dataset[ex_id][DATASET_HEADERS[args.dataset_nm]['summary']] if DATASET_HEADERS[args.dataset_nm]['summary'] is not None else '',
                     }
        if args.dataset_nm == 'wiki_bios':
            out_dict0["topic"] = re.fullmatch(r"Tell me a bio of (.*).", dataset[ex_id]['query']).group(1)
        if args.perturb_input2 is None:
            out_dict1 = {"input_index": ex_id,
                        "assigned_model": args.model_nm,
                        "assigned_process": 1,
                        "context_string": frags_before_article + frags_after_article,
                        "assigned_weight": -args.alpha,
                        "filter_p": 0.9,
                        "orig_id": dataset[ex_id]['id']
                        }
        else:
            out_dict1 = {"input_index": ex_id,
                        "assigned_model": args.model_nm,
                        "assigned_process": 1,
                        "context_string": frags_before_article + article_frag.format(shuffle_summ(dataset[ex_id][DATASET_HEADERS[args.dataset_nm]['source']], shuffle_type=args.perturb_input2)) + frags_after_article,
                        "assigned_weight": -args.alpha,
                        "filter_p": 0.9,
                        "orig_id": dataset[ex_id]['id']
                        }

        out_data.append(json.dumps(out_dict0))
        out_data.append(json.dumps(out_dict1))

    output_file_prefix, output_file_ext = os.path.splitext(args.output_file)
    output_file = f"{output_file_prefix}_{1.0 + args.alpha}_{-args.alpha}_{args.model_nm.replace('/','-')}_n{args.n_samples}_seed{args.shuffle_seed}{output_file_ext}"
    print(f"Writing output to: {output_file}")
    with open(output_file, 'w') as fout:
        fout.write('\n'.join(out_data))
