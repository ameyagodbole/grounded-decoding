import accelerate
import json
from datetime import datetime
import torch
from tqdm import trange
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from src.contrastive_llama import LlamaTokenTypeAttnForCausalLM

print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Imports done")

device = torch.device('cuda')
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

with open("../outputs/cohere_you_20nelsonqueries/search_results.json") as fin:
    all_queries = json.load(fin)

PROMPT_FORMAT = ["""[INST] <<SYS>> Write an fluent, informative essay responding to the given user query. Use the information in the following documents.\n""",
"""{}""",
"""<</SYS>>\n\n{} [/INST]"""]

ex_idx = 0
n_docs = 5

num_gpus = 2
kwargs = {"torch_dtype": torch.float16, "offload_folder": f"./llama-2-7b-chat-hf/offload", "device_map": "auto", "max_memory": {i: f"46GiB" for i in range(num_gpus)}}
model = LlamaTokenTypeAttnForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", **kwargs)

def tokenize_one_query(one_query, tokenizer, n_docs=10):
    formatted_search_results = '\n'.join([f"TITLE: {one_ret['title']}\nSNIPPET: {one_ret['snippet']}\n" for one_ret in one_query["search_results"][:n_docs]])
    tokenized_parts = tokenizer([PROMPT_FORMAT[0], '\n'+formatted_search_results, '\n' + PROMPT_FORMAT[2].format(one_query['query'])])
    full_input_ids = torch.LongTensor(tokenized_parts['input_ids'][0] + tokenized_parts['input_ids'][1][2:] + tokenized_parts['input_ids'][2][2:])[None, :]
    full_attn_mask = torch.LongTensor(tokenized_parts['attention_mask'][0] + tokenized_parts['attention_mask'][1][2:] + tokenized_parts['attention_mask'][2][2:])[None, :]
    context_mask = torch.LongTensor([0] * (len(tokenized_parts['input_ids'][0]) + 1) + [1] * (len(tokenized_parts['input_ids'][1][2:]) - 1) + [0] * len(tokenized_parts['input_ids'][2][2:]))[None, :]
    return {'input_ids': full_input_ids, 'attention_mask': full_attn_mask, 'context_attention_mask': context_mask}

all_predictions = {}
for ex_id in trange(len(all_queries[:4])):
    one_tokenized_query_prompt = tokenize_one_query(all_queries[ex_idx], tokenizer, n_docs=n_docs)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Running CAD")
    causal_output = model.generate(one_tokenized_query_prompt['input_ids'].to(device), 
                                num_beams=1, do_sample=False, top_p=None, temperature=None,
                                return_dict_in_generate=True, output_scores=True, output_logits=True,
                                context_attention_mask=one_tokenized_query_prompt['context_attention_mask'].to(device),
                                context_attention_weight=0.9, cad_alpha=0.0)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Running default")
    default_output = model.generate(one_tokenized_query_prompt['input_ids'].to(device),
                                    num_beams=1, do_sample=False, top_p=None, temperature=None,
                                    return_dict_in_generate=True, output_scores=True, output_logits=True)
    all_predictions[f"{ex_id}"] = {'query_prompt': tokenizer.batch_decode(one_tokenized_query_prompt['input_ids'])[0],
                                   'base_generation': tokenizer.batch_decode(default_output.sequences)[0],
                                   'cad_generation': tokenizer.batch_decode(causal_output.sequences)[0],
                                   }
with open('test_output.json', 'w') as fout:
    json.dump(all_predictions, fout, indent=4)