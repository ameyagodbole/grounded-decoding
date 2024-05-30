from accelerate import infer_auto_device_map, init_empty_weights
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from src.head_mask_llama import LlamaHeadMaskForCausalLM

prompt = """ <s> [INST]<<SYS>>
Summarize the article.
<</SYS>>

A bogus doctor who had 'more psychiatric qualifications than Sigmund Freud' and used her husband and parents as fake references has been unmasked as a serial liar. Sarah Sims claimed she had a long list of qualifications from the University of London and said she had been given a 'ladyship' by Warwick University. The three-times married 'doctor' also claimed she had been an officer in the Ministry of Defence as she applied to be head of childcare services at a children's home in Folkestone, Kent. But she was finally exposed as a fraud after the employers discovered her qualifications were completely made up. Sarah Sims claimed she had been given a doctorate by the University of London (pictured), but it just was one of a long list of lies she told . Sims, 41, claimed on her CV that she received a first class bachelor's degree, master's degree and philosophy doctorate from the University of London. She said her last job had been lead welfare officer at the MoD, and included glowing testimonial from her line manager Sergeant Major Darren Pugh. Three-times married Sims, who lives in a £1million home near Canterbury, also claimed to be a member of the 'British Association of Counselling and Psychology (BCAP).' She added that she had been awarded the title of 'Ladyship' from Warwick University for community 'good works'. The highly-qualified 'doctor' seemed the perfect fit for the role as operations director at Ferndearle Child Care Services - until it was revealed that nearly her entire CV was a lie. A jury at Canterbury Crown Court heard that Sims did not receive any of the psychology degrees she claimed to own, and her university had no record of her ever being a student there. Warwick University said it never gives out the title of 'ladyship' andthe British Association of Counselling and Psychology does not exist. Sims' glowing reference from her 'line manager' Sergeant Major Darren Pugh turned out be be her 48-year-old husband - Darren Pugh - who was himself cleared of fraud. The three-times married 'doctor' also claimed she had been an officer in the Ministry of Defence as she applied to be head of child care services at a nursing home, Canterbury Crown Court (pictured) heard . The court also heard that two other references listed on Sims' CV as 'therapeutic supervisors' were actually her parents. Piers Reed, prosecuting, told how Sims applied for the Folkestone post in February 2011, saying she was considering leaving the Army. Mr Reed said: 'She said she was a doctorof psychology and was interested in working for Fearndale. She contacted the manager and part-owner of the home and arranged to visit.' 'Mr Pugh had been in the Army, but not at that time. In her job application, she stated she had been a student at London University and had obtained a BA in Psychology, an MA and Phd in Behavioural Psychology. 'Under the heading, she said she was a member of BCAP and, under the Awards heading, stated that,in May 1997, she had been awarded a ladyship from Warwick University for services to the community.' But Mr Reed said that said Sims appeared to have 'more psychiatric qualifications than Sigmund Freud', but there were no records of her attending university. Mr Reed added: 'BCAP doesn't exist. The British Psychological Society and the British Association of Counselling and Psychotherapy do exist, but neither have ever heard of her. 'And the University of Warwick, while conferring awards, never confer titles such as 'Ladyship'. 'In her CV, she also claimed to have worked as a lead consultant director for a company called EIS. There are no records of her ever working for EIS.' Mr Reed told the court that Sims obtained a mortgage onher property after telling Halifax she was earning between £80,000 and £150,000. In fact, at the time, she was jobless. Sims was found guilty of four counts of fraud and will be sentenced next month.[/INST]Summary:"""

accl_args = {"attn_implementation": "sdpa",
             "device_map": 'auto',
             "offload_folder": f"/tmp/ameya/2024/offload/debug/",
             "max_memory": {i: f"46GiB" for i in range(2)}
             }

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
inputs = tokenizer(prompt, return_tensors='pt')

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", **accl_args)
output = model.generate(inputs['input_ids'].to('cuda'), top_p=0.9)

del model

block_list = [[16, 19], [11, 15], [8, 26], [6, 9], [7, 12], [17, 22], [11, 2], [6, 16], [19, 15], [21, 30],
              [18, 30], [24, 29], [7, 4], [15, 14], [16, 1], [20, 30], [17, 0], [14, 18], [24, 30], [21, 1]]
model = LlamaHeadMaskForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", **accl_args)
output = model.generate(inputs['input_ids'].to('cuda'), top_p=0.9,  cad_alpha=1.0,
                        cad_kv_sharing='total', block_list=block_list)
