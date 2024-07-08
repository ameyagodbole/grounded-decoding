import torch
import numpy as np
from src.head_mask_llama import LlamaSdpaAttention, LlamaAttention

from transformers import AutoTokenizer, AutoConfig

config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

base_layer = LlamaAttention(config, layer_idx=0)
sdpa_layer = LlamaSdpaAttention(config, layer_idx=0)

sdpa_layer.load_state_dict(base_layer.cpu().state_dict())

base_layer.eval()
sdpa_layer.eval()
base_layer = base_layer.to(torch.device('cuda'))
sdpa_layer = sdpa_layer.to(torch.device('cuda'))

sample_input = torch.FloatTensor(np.random.randn(2, 20, config.hidden_size))
sample_input = sample_input.to('cuda')

with torch.no_grad():
    base_output = base_layer(sample_input, position_ids=torch.arange(sample_input.shape[1])[None, :].cuda(), output_attentions=True)
    sdpa_output = sdpa_layer(sample_input, position_ids=torch.arange(sample_input.shape[1])[None, :].cuda())

value_diff = (base_output[0] - sdpa_output[0]).max().cpu().item()
print(f"Eager vs SDPA value diff: {value_diff}")

dtype, device = sample_input.dtype, sample_input.device
min_dtype = torch.finfo(dtype).min
sequence_length = sample_input.shape[1]
target_length = sequence_length
causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
if sequence_length != 1:
    causal_mask = torch.triu(causal_mask, diagonal=1)
causal_mask = causal_mask[None, None, :, :].expand(sample_input.shape[0], 1, -1, -1)

attn_mask = torch.FloatTensor([[1]*15 + [0]*5, [1]*18 + [0]*2]).to('cuda')
ctxt_mask = torch.FloatTensor([[0]*5 + [1]*5 + [0]*10, [0]*6 + [1]*8 + [0]*6]).to('cuda')

causal_mask1 = causal_mask.clone()
mask_length = attn_mask.shape[-1]
padding_mask1 = causal_mask1[..., :mask_length].eq(0.0) * attn_mask[:, None, None, :].eq(0.0)
causal_mask1[..., :mask_length] = causal_mask1[..., :mask_length].masked_fill(padding_mask1, min_dtype)

print("Diff between causal_mask and causal_mask + attn_mask")
print(torch.vstack(torch.where(causal_mask[0, 0] != causal_mask1[0, 0])))

causal_mask2 = causal_mask1.clone()
mask_length = ctxt_mask.shape[-1]
padding_mask2 = causal_mask2[..., :mask_length].eq(0.0) * ctxt_mask[:, None, None, :].eq(1.0)
cad_beta = torch.log(torch.FloatTensor([0.9])).item()
causal_mask2[..., :mask_length] = causal_mask2[..., :mask_length].masked_fill(padding_mask2, cad_beta)

print("causal_mask2[0, 0, 10]:")
print(causal_mask2[0, 0, 10])

print("Diff between causal_mask and causal_mask + attn_mask + context_mask")
print(torch.vstack(torch.where(causal_mask[0, 0] != causal_mask2[0, 0])))

print("Diff between causal_mask + attn_mask and causal_mask + attn_mask + context_mask")
print(torch.vstack(torch.where(causal_mask1[0, 0] != causal_mask2[0, 0])))

with torch.no_grad():
    base_output1 = base_layer(sample_input, position_ids=torch.arange(sample_input.shape[1])[None, :].cuda(), 
                              attention_mask=causal_mask1, output_attentions=True)
    base_output2 = base_layer(sample_input, position_ids=torch.arange(sample_input.shape[1])[None, :].cuda(), 
                              attention_mask=causal_mask1, output_attentions=True, 
                              block_list=[[0, 1], [0,5], [1,0]])

print(f"Causal attn:\n{base_output1[1][0, 1, 10]}")
print(f"Causal_mask + Attn_mask:\n{base_output2[1][0, 1, 10]}")

for h_id in range(32):
    if h_id not in (1, 5,):
        assert (base_output1[1][0, h_id] == base_output2[1][0, h_id]).all()

with torch.no_grad():
    base_output3 = base_layer(sample_input, position_ids=torch.arange(sample_input.shape[1])[None, :].cuda(), 
                              attention_mask=causal_mask1, output_attentions=True, 
                              block_list=[[0, 1], [0,5], [1,0]], context_causal_mask=causal_mask2)
print(f"Causal_mask + Attn_mask + Context_mask:\n{base_output3[1][0, 1, 10]}")

for h_id in range(32):
    if h_id not in (1, 5,):
        assert (base_output1[1][0, h_id] == base_output3[1][0, h_id]).all()

temp_base_output1 = base_output1[1][0, 1, 10].clone()
temp_base_output1[ctxt_mask[0].to(torch.bool)] *= torch.FloatTensor([0.9]).cuda()
temp_base_output1 /= temp_base_output1.sum()

print("Verify difference between explicit computation and layer execution matches:")
print(torch.abs(temp_base_output1 - base_output3[1][0, 1, 10]).max().item())

with torch.no_grad():
    sdpa_output1 = sdpa_layer(sample_input, position_ids=torch.arange(sample_input.shape[1])[None, :].cuda(), 
                              attention_mask=causal_mask1, output_attentions=False)
    sdpa_output2 = sdpa_layer(sample_input, position_ids=torch.arange(sample_input.shape[1])[None, :].cuda(), 
                              attention_mask=causal_mask1, output_attentions=False, 
                              block_list=[[0, 1], [0,5], [1,0]])
    sdpa_output3 = sdpa_layer(sample_input, position_ids=torch.arange(sample_input.shape[1])[None, :].cuda(),
                              attention_mask=causal_mask1, output_attentions=False,
                              block_list=[[0, 1], [0,5], [1,0]], context_causal_mask=causal_mask2)

print("Diff between eager vs causal (causal_mask)")
print(torch.abs(base_output1[0] - sdpa_output1[0]).max().item())

print("Diff between eager vs causal (causal_mask + attn_mask)")
print(torch.abs(base_output2[0] - sdpa_output2[0]).max().item())

print("Diff between eager vs causal (causal_mask + attn_mask + context_mask)")
print(torch.abs(base_output3[0] - sdpa_output3[0]).max().item())

print("Diff between eager vs causal (causal_mask + attn_mask + context_mask)")
for _ in range(20):
    with torch.no_grad():
        new_sample_input = torch.FloatTensor(np.random.randn(2, 20, config.hidden_size))
        new_sample_input = new_sample_input.to('cuda')
        base_output_tmp = base_layer(new_sample_input, position_ids=torch.arange(new_sample_input.shape[1])[None, :].cuda(),
                                     attention_mask=causal_mask1, output_attentions=True,
                                     block_list=[[0, 1], [0,5], [1,0]], context_causal_mask=causal_mask2)
        print('---')
        sdpa_output_tmp = sdpa_layer(new_sample_input, position_ids=torch.arange(new_sample_input.shape[1])[None, :].cuda(),
                                    attention_mask=causal_mask1, output_attentions=False,
                                    block_list=[[0, 1], [0,5], [1,0]], context_causal_mask=causal_mask2)
        print(torch.abs(base_output_tmp[0] - sdpa_output_tmp[0]).max().item())
        print('+++')

print("+++++++++++\nTEST FALLBACK\n+++++++++++\n")
print("Diff between eager vs fallback from sdpa")
for _ in range(20):
    with torch.no_grad():
        new_sample_input = torch.FloatTensor(np.random.randn(2, 20, config.hidden_size))
        new_sample_input = new_sample_input.to('cuda')
        base_output_tmp = base_layer(new_sample_input, position_ids=torch.arange(new_sample_input.shape[1])[None, :].cuda(),
                                     attention_mask=causal_mask1, output_attentions=True,
                                     block_list=[[0, 1], [0,5], [1,0]], context_causal_mask=causal_mask2)
        print('---')
        sdpa_output_tmp = sdpa_layer(new_sample_input, position_ids=torch.arange(new_sample_input.shape[1])[None, :].cuda(),
                                    attention_mask=causal_mask1, output_attentions=True,
                                    block_list=[[0, 1], [0,5], [1,0]], context_causal_mask=causal_mask2)
        print(torch.abs(base_output_tmp[0] - sdpa_output_tmp[0]).max().item())
        print('+++')


import json
from src.head_mask_llama import LlamaHeadMaskForCausalLM

num_gpus = torch.cuda.device_count()
attn_impl = "sdpa"
accl_args = {"attn_implementation": attn_impl,
             "device_map": "auto",
             "offload_folder": f"/tmp/ameya/2024/offload/",
             "max_memory": {i: f"46GiB" for i in range(num_gpus)},
             "torch_dtype": torch.float16
             }

model = LlamaHeadMaskForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", **accl_args)
model.eval()

with open("configs/retrieval_head_score/llama-2-7b-80k.json", "r") as file:
    stable_block_list =  json.loads(file.readline())
stable_block_list = [(l[0], np.mean(l[1])) for l in stable_block_list.items()]
stable_block_list = sorted(stable_block_list, key=lambda x: x[1], reverse=True)
block_list = [[int(ll) for ll in l[0].split("-")] for l in stable_block_list][:50]

input_str = """[INST]<<SYS>>
Summarize the article.
<</SYS>>

A bogus doctor who had 'more psychiatric qualifications than Sigmund Freud' and used her husband and parents as fake references has been unmasked as a serial liar. Sarah Sims claimed she had a long list of qualifications from the University of London and said she had been given a 'ladyship' by Warwick University. The three-times married 'doctor' also claimed she had been an officer in the Ministry of Defence as she applied to be head of childcare services at a children's home in Folkestone, Kent. But she was finally exposed as a fraud after the employers discovered her qualifications were completely made up. Sarah Sims claimed she had been given a doctorate by the University of London (pictured), but it just was one of a long list of lies she told . Sims, 41, claimed on her CV that she received a first class bachelor's degree, master's degree and philosophy doctorate from the University of London. She said her last job had been lead welfare officer at the MoD, and included glowing testimonial from her line manager Sergeant Major Darren Pugh. Three-times married Sims, who lives in a £1million home near Canterbury, also claimed to be a member of the 'British Association of Counselling and Psychology (BCAP).' She added that she had been awarded the title of 'Ladyship' from Warwick University for community 'good works'. The highly-qualified 'doctor' seemed the perfect fit for the role as operations director at Ferndearle Child Care Services - until it was revealed that nearly her entire CV was a lie. A jury at Canterbury Crown Court heard that Sims did not receive any of the psychology degrees she claimed to own, and her university had no record of her ever being a student there. Warwick University said it never gives out the title of 'ladyship' and the British Association of Counselling and Psychology does not exist. Sims' glowing reference from her 'line manager' Sergeant Major Darren Pugh turned out be be her 48-year-old husband - Darren Pugh - who was himself cleared of fraud. The three-times married 'doctor' also claimed she had been an officer in the Ministry of Defence as she applied to be head of child care services at a nursing home, Canterbury Crown Court (pictured) heard . The court also heard that two other references listed on Sims' CV as 'therapeutic supervisors' were actually her parents.
 Piers Reed, prosecuting, told how Sims applied for the Folkestone post in February 2011, saying she was considering leaving the Army. Mr Reed said: 'She sa
id she was a doctor of psychology and was interested in working for Fearndale. She contacted the manager and part-owner of the home and arranged to visit.' 'Mr Pugh had been in the Army, but not at that time. In her job application, she stated she had been a student at London University and had obtained a BA in
 Psychology, an MA and Phd in Behavioural Psychology. 'Under the heading, she said she was a member of BCAP and, under the Awards heading, stated that, in May 1997, she had been awarded a ladyship from Warwick University for services to the community.' But Mr Reed said that said Sims appeared to have 'more psychiatric qualifications than Sigmund Freud', but there were no records of her attending university. Mr Reed added: 'BCAP doesn't exist. The British Psychological Society and the British Association of Counselling and Psychotherapy do exist, but neither have ever heard of her. 'And the University of Warwick, whil
e conferring awards, never confer titles such as 'Ladyship'. 'In her CV, she also claimed to have worked as a lead consultant director for a company called EIS. There are no records of her ever working for EIS.' Mr Reed told the court that Sims obtained a mortgage on her property after telling Halifax she was earning between £80,000 and £150,000. In fact, at the time, she was jobless. Sims was found guilty of four counts of fraud and will be sentenced next month.
[/INST]Summary:"""

input_ids = tokenizer(input_str)


# print("+++++++++++\nTEST WITHOUT ATTN MASK\n+++++++++++\n")

# attn_mask2 = torch.FloatTensor([[1]*20, [1]*20]).to('cuda')

# causal_mask1_new = causal_mask.clone()
# mask_length = attn_mask2.shape[-1]
# padding_mask = causal_mask1_new[..., :mask_length].eq(0.0) * attn_mask2[:, None, None, :].eq(0.0)
# causal_mask1_new[..., :mask_length] = causal_mask1_new[..., :mask_length].masked_fill(padding_mask, min_dtype)

# causal_mask2_new = causal_mask1_new.clone()
# mask_length = ctxt_mask.shape[-1]
# padding_mask2_new = causal_mask2_new[..., :mask_length].eq(0.0) * ctxt_mask[:, None, None, :].eq(1.0)
# cad_beta = torch.log(torch.FloatTensor([0.9])).item()
# causal_mask2_new[..., :mask_length] = causal_mask2_new[..., :mask_length].masked_fill(padding_mask2_new, cad_beta)

# for _ in range(10):
#     with torch.no_grad():
#         new_sample_input = torch.FloatTensor(np.random.randn(2, 20, config.hidden_size))
#         new_sample_input = new_sample_input.to('cuda')
#         base_output_tmp1 = base_layer(new_sample_input, position_ids=torch.arange(new_sample_input.shape[1])[None, :].cuda(),
#                                       attention_mask=causal_mask1_new, output_attentions=True,
#                                       block_list=[[0, 1], [0,5], [1,0]], context_causal_mask=causal_mask2_new)
#         base_output_tmp2 = base_layer(new_sample_input, position_ids=torch.arange(new_sample_input.shape[1])[None, :].cuda(),
#                                       output_attentions=True,
#                                       block_list=[[0, 1], [0,5], [1,0]], context_causal_mask=causal_mask2_new)
#         print(torch.abs(base_output_tmp1[0] - base_output_tmp2[0]).max().item())
#         print('---')
#         sdpa_output_tmp1 = sdpa_layer(new_sample_input, position_ids=torch.arange(new_sample_input.shape[1])[None, :].cuda(),
#                                       attention_mask=causal_mask1_new, output_attentions=False,
#                                       block_list=[[0, 1], [0,5], [1,0]], context_causal_mask=causal_mask2_new)
#         sdpa_output_tmp2 = sdpa_layer(new_sample_input, position_ids=torch.arange(new_sample_input.shape[1])[None, :].cuda(),
#                                       output_attentions=False,
#                                       block_list=[[0, 1], [0,5], [1,0]], context_causal_mask=causal_mask2_new)
#         print(torch.abs(sdpa_output_tmp1[0] - sdpa_output_tmp2[0]).max().item())
#         print('+++')

