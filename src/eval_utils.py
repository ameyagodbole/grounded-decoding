import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


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