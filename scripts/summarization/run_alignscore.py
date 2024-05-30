from itertools import chain
import json
import numpy as np
import os
import wandb
from alignscore import AlignScore


scorer = AlignScore(model='roberta-large', batch_size=32, device='cuda:0',
                    ckpt_path='/home/ameya/AlignScore/ckpts/AlignScore-large.ckpt', evaluation_mode='nli_sp')

api = wandb.Api()
# cnn-dm
# regular_runs = [api.run(f'ameyag416/grounded-decoding/{r_id}') for r_id in ["x3oezgn9", "xs45ymzb", "q43nxgdt"]]
# sweep = api.sweep("ameyag416/grounded-decoding/c9yzgor1")
# sweep_runs = [sweep.runs]
# context_keynm = 'article'

# cnn-dm + fp16
regular_runs = [api.run(f'ameyag416/grounded-decoding/{r_id}') for r_id in ["uhsf5oiz", "222xaqh2", "9py04wmy"]]
sweeps = [api.sweep("ameyag416/grounded-decoding/zmbae8vt"),]
sweep_runs = [sweep.runs for sweep in sweeps]
context_keynm = 'article'

# xsum
# regular_runs = [api.run(f'ameyag416/grounded-decoding/{r_id}') for r_id in ["o8jr3zzc", "ual7yaow", "vwtwfhmn"]]
# sweeps = [api.sweep("ameyag416/grounded-decoding/58u6uyd7"), api.sweep("ameyag416/grounded-decoding/gmqqc4u6")]
# sweep_runs = [sweep.runs for sweep in sweeps]
# context_keynm = 'document'

for run in chain(regular_runs, *sweep_runs):
    print(f"Processing run: {run.name} ({run.id})")
    if run.state != 'finished':
        print(f"Incomplete run: {run.name} ({run.id})")
        print('+++++')
        continue

    output_files = os.listdir(run.config['output_dir'])
    if len(output_files) > 1:
        output_file = f"""{run.config['model_nm']}-{run.config['decoding_algo']}{'-topk' + str(run.config['topk']) if run.config['topk'] is not None else ''}{f"-topp{float(run.config['topp'])}" if run.config['topp'] is not None else ''}{f"-temp{float(run.config['temperature'])}" if run.config['temperature'] is not None else ''}{f"-alpha{float(run.config['cad_alpha'])}" if run.config['cad_alpha'] is not None else ''}{'-mask_topk' + str(run.config['mask_topk']) if run.config['mask_topk'] is not None else ''}-n{run.config['n_samples'] or ''}-r{run.config['shuffle_seed'] or ''}.jsonl"""
        assert output_file in output_files
        output_file = os.path.join(run.config['output_dir'], output_file)
    elif 'alignscore' in run.summary:
        print("AlignScore already computed")
        print('+++++')
        continue
    else:
        output_file = os.path.join(run.config['output_dir'], output_files[0])
    print(f"Processing file: {output_file}")
    if not os.path.exists(output_file):
        print("Missing output file")
        print('+++++')
        continue

    model_preds = []
    with open(output_file) as fin:
        for line in fin:
            model_preds.append(json.loads(line))
    
    articles, predictions = [], []
    for pred in model_preds:
        articles.append(pred[context_keynm])
        predictions.append(pred['model_prediction'].replace('<span>', '').replace('</span>', '').replace('</s>', '').strip())

    scores = scorer.score(contexts=articles, claims=predictions)
    run.summary['alignscore'] = np.mean(scores)
    run.summary.update()
    
    print('+++++')
