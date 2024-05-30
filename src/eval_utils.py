import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# from alignscore import AlignScore


def compute_factkb_score(predictions, articles, model_kwargs=None, batch_size=16):
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


# def compute_alignscore(predictions, articles):
#     with torch.no_grad():
#         scorer = AlignScore(model='roberta-large', batch_size=32, device='cuda:0',
#                             ckpt_path='/path/to/checkpoint',
#                             evaluation_mode='nli_sp')
#         score = scorer.score(contexts=articles, claims=predictions)
