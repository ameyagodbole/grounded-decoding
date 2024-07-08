import argparse
from itertools import chain
import json
import numpy as np
import os
import wandb
from alignscore import AlignScore


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", action='store_true')
    parser.add_argument("--pred_file", type=str)
    parser.add_argument("--data_file", type=str)
    args = parser.parse_args()

    scorer = AlignScore(model='roberta-large', batch_size=32, device='cuda:0',
                        ckpt_path='/home/ameya/AlignScore/ckpts/AlignScore-large.ckpt', evaluation_mode='nli_sp')
    
    if args.wandb:
        api = wandb.Api()
        # cnn-dm
        # regular_runs = [api.run(f'ameyag416/grounded-decoding/{r_id}') for r_id in ["x3oezgn9", "xs45ymzb", "q43nxgdt"]]
        # sweep = api.sweep("ameyag416/grounded-decoding/c9yzgor1")
        # sweep_runs = [sweep.runs]
        # context_keynm = 'article'

        # cnn-dm + fp16
        # regular_runs = [api.run(f'ameyag416/grounded-decoding/{r_id}') for r_id in ["uhsf5oiz", "222xaqh2", "9py04wmy"]]
        # sweeps = [api.sweep("ameyag416/grounded-decoding/zmbae8vt"),]
        # sweep_runs = [sweep.runs for sweep in sweeps]
        # context_keynm = 'article'

        # cnn-dm + ragtruthprompt
        # regular_runs = [api.run(f'ameyag416/grounded-decoding/{r_id}') for r_id in ["htdnxbyb", "ar567caq"]]
        # sweep_runs = []
        # context_keynm = 'article'

        # xsum
        # regular_runs = [api.run(f'ameyag416/grounded-decoding/{r_id}') for r_id in ["o8jr3zzc", "ual7yaow", "vwtwfhmn"]]
        # sweeps = [api.sweep("ameyag416/grounded-decoding/58u6uyd7"), api.sweep("ameyag416/grounded-decoding/gmqqc4u6")]
        # sweep_runs = [sweep.runs for sweep in sweeps]
        # context_keynm = 'document'

        # cnn-dm + xsum
        # regular_runs = []
        # sweeps = [api.sweep("ameyag416/grounded-decoding/kmnz5ksa"), api.sweep("ameyag416/grounded-decoding/2j09zr2t")]
        # sweep_runs = [sweep.runs for sweep in sweeps]

        # wiki_bios
        # regular_runs = [api.run(f'ameyag416/grounded-decoding/{r_id}') for r_id in ["mfjgs2h5"]]
        # sweeps = [api.sweep("ameyag416/grounded-decoding/e1gwe0yw"), api.sweep("ameyag416/grounded-decoding/jxbqaitn")]
        # sweep_runs = [sweep.runs for sweep in sweeps]

        # ragtruth_news
        regular_runs = [api.run(f'ameyag416/grounded-decoding/{r_id}') for r_id in ["j4ybm6xa", "kysug9xu", "vgm4jbb0"]]
        # ["4wgnut6p", "1d0sdb0f", "wkmt4uzk"]
        # ["j4ybm6xa", "kysug9xu", "vgm4jbb0"]

        # regular_runs = []
        # sweeps = [api.sweep("ameyag416/grounded-decoding/bu9snjv8"), api.sweep("ameyag416/grounded-decoding/5nnuvv19"), api.sweep("ameyag416/grounded-decoding/7fqbfg5r")]
        # sweeps = [api.sweep("ameyag416/grounded-decoding/7fqbfg5r")]
        sweeps = []
        sweep_runs = [sweep.runs for sweep in sweeps]

        for run in chain(regular_runs, *sweep_runs):
            if 'xsum' in run.tags:
                context_keynm = 'document'
            elif 'cnn_dailymail' in run.tags:
                context_keynm = 'article'
            elif 'wiki_bios' in run.tags:
                context_keynm = 'user_message'
            elif 'ragtruth_news' in run.tags:
                context_keynm = 'source_info'
            else:
                raise NotImplementedError(f"Unknown context_keynm for run with tags: {run.tags}")

            print(f"Processing run: {run.name} ({run.id})")
            if run.state != 'finished':
                print(f"Incomplete run: {run.name} ({run.id})")
                print('+++++')
                continue

            output_files = os.listdir(run.config['output_dir'])
            if len(output_files) > 1:
                if run.config['decoding_algo'] == 'regular':
                    output_file = f"""{run.config['model_nm']}-{run.config['decoding_algo']}-n{run.config['n_samples'] or ''}-r{run.config['shuffle_seed'] or ''}.jsonl"""
                else:
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
                if context_keynm == 'user_message':
                    full_context = pred[context_keynm]
                    assert full_context[-len(pred['query']):] == pred['query']
                    full_context = full_context[:-len(pred['query'])].strip()
                    articles.append(full_context)
                else:
                    articles.append(pred[context_keynm])
                predictions.append(pred['model_prediction'].replace('<span>', '').replace('</span>', '').replace('</s>', '').strip())

            scores = scorer.score(contexts=articles, claims=predictions)
            run.summary['alignscore'] = np.mean(scores)
            run.summary.update()
            
            print('+++++')
    elif args.pred_file and args.data_file:
        context_keynm = 'article'

        print("Processing file:")
        print(f"Pred: {args.pred_file}")
        print(f"Data: {args.data_file}")
        if not os.path.exists(args.pred_file):
            print("Missing prediction file")
            print('+++++')
            exit(0)
        if not os.path.exists(args.data_file):
            print("Missing data file")
            print('+++++')
            exit(0)
        
        model_preds = []
        with open(args.pred_file) as fin:
            for line in fin:
                model_preds.append(json.loads(line))
        input_data = []
        with open(args.data_file) as fin:
            for line in fin:
                one_ex = json.loads(line)
                if one_ex['assigned_process'] == 0:
                    input_data.append(one_ex)

        articles, predictions = [], []
        for pred in model_preds:
            pred_str = pred["string"][0].replace('<span>', '').replace('</span>', '').replace('</s>', '').strip()
            predictions.append(pred_str)
            articles.append(input_data[pred['input_index']][context_keynm])

        scores = scorer.score(contexts=articles, claims=predictions)
        print(f"len(scores): {len(scores)}")
        print(f'alignscore: {np.mean(scores)}')
        print('+++++')
    else:
        print("Nothing to do")
