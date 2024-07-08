import logging
import numpy as np
import torch
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from minicheck.minicheck import MiniCheck
# from alignscore import AlignScore

logger = logging.getLogger(__name__)


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


def compute_minicheck_score(predictions, articles, scorer_nm='flan-t5-large', use_cuda=True):
    def _split_predictions_to_sents(_predictions, _articles):
        sents_added = 0
        split_map = {}
        _prediction_sents, _articles_copied = [], []
        for ex_ctr, (one_pred, one_article) in enumerate(zip(_predictions, _articles)):
            one_pred_blocks = one_pred.split('\n')
            one_pred_sents = []
            for one_blk in one_pred_blocks:
                if len(one_blk) > 0:
                    one_pred_sents.extend(sent_tokenize(one_blk))
            split_map[ex_ctr] = list(range(sents_added,sents_added+len(one_pred_sents)))
            sents_added += len(one_pred_sents)
            _prediction_sents.extend(one_pred_sents)
            _articles_copied.extend([one_article]*len(one_pred_sents))
            assert sents_added == len(_prediction_sents) == len(_articles_copied)
        logger.info(f"Expanded {len(_predictions)} examples to {len(_prediction_sents)} claim sentences.")
        return split_map, _prediction_sents, _articles_copied

    def _aggregate_metrics(labels_pred, prob_pred, ex2idx, _prediction_sents, _articles_copied):
        # pred_prob is currently unused
        labels_pred_np = np.array(labels_pred)
        strict_sum, macro_sum, micro_sum = 0, 0, 0
        strict_total, macro_total, micro_total = 0, 0, 0
        pred_log = []
        for _, claim_idx in ex2idx.items():
            strict_total += 1
            macro_total += 1
            micro_total += len(claim_idx)
            ex_preds = labels_pred_np[claim_idx]
            one_log = [{'label': labels_pred[cl_idx], 'prob': prob_pred[cl_idx],
                        'claim': _prediction_sents[cl_idx]} for cl_idx in claim_idx]
            assert len(set(_articles_copied[cl_idx] for cl_idx in claim_idx)) == 1
            pred_log.append({'article': _articles_copied[claim_idx[0]], 'predictions': one_log})
            strict_sum += float(np.all(ex_preds))
            macro_sum += float(np.mean(ex_preds))
            micro_sum += float(np.sum(ex_preds))
        metrics = {'strict_score': strict_sum / strict_total,
                   'macro_score': macro_sum / macro_total,
                   'micro_score': micro_sum / micro_total
        }
        return metrics, pred_log

    scorer = MiniCheck(model_name=scorer_nm, device='cuda:0' if use_cuda else 'cpu', cache_dir='./ckpts')
    split_idx, prediction_sents, articles_copied = _split_predictions_to_sents(predictions, articles)
    pred_label, raw_prob, _, _ = scorer.score(docs=articles_copied, claims=prediction_sents)
    metrics, metrics_log = _aggregate_metrics(pred_label, raw_prob, split_idx, prediction_sents, articles_copied)
    del scorer
    return metrics, metrics_log


# def compute_alignscore(predictions, articles):
#     with torch.no_grad():
#         scorer = AlignScore(model='roberta-large', batch_size=32, device='cuda:0',
#                             ckpt_path='/path/to/checkpoint',
#                             evaluation_mode='nli_sp')
#         score = scorer.score(contexts=articles, claims=predictions)
