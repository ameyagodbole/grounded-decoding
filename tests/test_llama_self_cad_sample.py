import accelerate
import json
# import os
from datetime import datetime
import torch
# from transformers.integrations import HfDeepSpeedConfig
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
# from contrastive_llama import LlamaTokenTypeAttnForCausalLM
from src.contrastive_llama import LlamaTokenTypeAttnForCausalLM
# import deepspeed

print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Imports done")

# distributed setup
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# local_rank = int(os.getenv("LOCAL_RANK", "0"))
# world_size = int(os.getenv("WORLD_SIZE", "1"))
# torch.cuda.set_device(local_rank)
# deepspeed.init_distributed()

# config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
# model_hidden_size = config.d_model

# ds_config = {
#     "fp16": {
#         "enabled": False
#     },
#     "bf16": {
#         "enabled": False
#     },
#     "zero_optimization": {
#         "stage": 3,
#         "offload_param": {
#             "device": "cpu",
#             "pin_memory": True
#         },
#         "overlap_comm": True,
#         "contiguous_gradients": True,
#         "reduce_bucket_size": model_hidden_size * model_hidden_size,
#         "stage3_prefetch_bucket_size": 0.9 * model_hidden_size * model_hidden_size,
#         "stage3_param_persistence_threshold": 10 * model_hidden_size
#     },
#     "steps_per_print": 2000,
#     "train_batch_size": 1,
#     "train_micro_batch_size_per_gpu": 1,
#     "wall_clock_breakdown": False
# }
# dschf = HfDeepSpeedConfig(ds_config)

device = torch.device('cuda')
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

with open("outputs/cohere_you_20nelsonqueries/search_results.json") as fin:
    all_queries = json.load(fin)

PROMPT_FORMAT = ["""[INST] <<SYS>> Write an fluent, informative essay responding to the given user query. Use the information in the following documents.\n""",
"""{}""",
"""<</SYS>>\n\n{} [/INST]"""]

ex_idx = 0
n_docs = 5

# # TEST 1
# # Compare greedy decoding of AutoModelForCausalLM to LlamaTokenTypeAttnForCausalLM
# one_query = all_queries[ex_idx]
# formatted_search_results = '\n'.join([f"TITLE: {one_ret['title']}\nSNIPPET: {one_ret['snippet']}\n" for one_ret in one_query["search_results"][:n_docs]])
# one_query_prompt = '\n'.join(PROMPT_FORMAT).format(formatted_search_results, all_queries[ex_idx]['query'])

# base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
# base_model = base_model.to(device)
# base_output = base_model.generate(tokenizer(one_query_prompt, return_tensors='pt')['input_ids'].to(device), 
#                                   num_beams=1, do_sample=False, top_p=None, temperature=None, 
#                                   return_dict_in_generate=True, output_scores=True, output_logits=True, output_attentions=True)
# del base_model

num_gpus = 1
kwargs = {"offload_folder": f"./llama-2-7b-chat-hf/offload", "device_map": "auto", "max_memory": {i: f"46GiB" for i in range(num_gpus)}}
model = LlamaTokenTypeAttnForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", **kwargs)
# model = model.to(device)
# model = model.to(dtype=torch.float16, device=device)

# causal_output = model.generate(tokenizer(one_query_prompt, return_tensors='pt')['input_ids'].to(device), 
#                                num_beams=1, do_sample=False, top_p=None, temperature=None,
#                                return_dict_in_generate=True, output_scores=True, output_logits=True, output_attentions=True)

# # Compare
# assert all([torch.all(bl == cl) for bl, cl in zip(base_output.logits, causal_output.logits)])
# assert all([torch.all(bl == cl) for bl, cl in zip(base_output.scores, causal_output.scores)])
# assert torch.all(base_output.attentions[0] == causal_output.attentions[0])

# TEST 2
# Compare logits with and without context masking
def tokenize_one_query(one_query, tokenizer, n_docs=10):
    formatted_search_results = '\n'.join([f"TITLE: {one_ret['title']}\nSNIPPET: {one_ret['snippet']}\n" for one_ret in one_query["search_results"][:n_docs]])
    tokenized_parts = tokenizer([PROMPT_FORMAT[0], '\n'+formatted_search_results, '\n' + PROMPT_FORMAT[2].format(one_query['query'])])
    full_input_ids = torch.LongTensor(tokenized_parts['input_ids'][0] + tokenized_parts['input_ids'][1][2:] + tokenized_parts['input_ids'][2][2:])[None, :]
    full_attn_mask = torch.LongTensor(tokenized_parts['attention_mask'][0] + tokenized_parts['attention_mask'][1][2:] + tokenized_parts['attention_mask'][2][2:])[None, :]
    context_mask = torch.LongTensor([0] * (len(tokenized_parts['input_ids'][0]) + 1) + [1] * (len(tokenized_parts['input_ids'][1][2:]) - 1) + [0] * len(tokenized_parts['input_ids'][2][2:]))[None, :]
    return {'input_ids': full_input_ids, 'attention_mask': full_attn_mask, 'context_attention_mask': context_mask}

one_tokenized_query_prompt = tokenize_one_query(all_queries[ex_idx], tokenizer, n_docs=n_docs)
print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Running CAD")
import pdb; pdb.set_trace()
causal_output = model.generate(one_tokenized_query_prompt['input_ids'].to(device), 
                               num_beams=1, do_sample=True, top_p=0.9, temperature=1.0,
                               return_dict_in_generate=True, output_scores=True, output_logits=True, output_attentions=True,
                               context_attention_mask=one_tokenized_query_prompt['context_attention_mask'].to(device),
                               context_attention_weight=0.9, cad_alpha=8.0)

causal_output = model.generate(one_tokenized_query_prompt['input_ids'].to(device), 
                               num_beams=1, do_sample=False, top_p=None, temperature=None,
                               return_dict_in_generate=True, output_scores=True, output_logits=True, output_attentions=True,
                               context_attention_mask=one_tokenized_query_prompt['context_attention_mask'].to(device),
                               context_attention_weight=0.9, cad_alpha=8.0)
# print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Running default")
# default_output = model.generate(one_tokenized_query_prompt['input_ids'].to(device),
#                                 num_beams=1, do_sample=False, top_p=None, temperature=None,
#                                 return_dict_in_generate=True, output_scores=True, output_logits=True, output_attentions=True)

# assert all([torch.all(bl == cl) for bl, cl in zip(default_output.scores, causal_output.scores)])
# assert all([torch.all(bl == cl) for bl, cl in zip(default_output.logits, causal_output.base_logits)])
