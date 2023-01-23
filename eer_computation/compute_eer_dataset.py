import os
import subprocess
from pathlib import Path

data_root = Path('/storage/brno2/home/deemax/IDSD_new/score')

train_datasets = ['F2S', 'FREC', 'AS19', 'WF']
# train_datasets = ['AS19']
eval_datasets = ['F2S', 'FREC', 'AS19', 'WF', 'AVC', 'FM', 'AS21']
features = ['mel', 'stft', 'cqt', 'vqt', 'iirt', 'mfcc', 'chroma']

gen_scores = {}
df_scores = {}

for feature in features:
    gen_scores[feature] = []
    df_scores[feature] = []

for train_dataset in train_datasets:
    for eval_dataset in eval_datasets:
        if train_dataset == eval_dataset:
            continue

        data_folder = f'{train_dataset}-{eval_dataset}'
        # data_folder = f'{train_dataset}'

        for feature in features:
            gen_score = data_root.joinpath(data_folder).joinpath(f'{data_folder}-eval-{feature}-eval-genuine.txt')
            df_score = data_root.joinpath(data_folder).joinpath(f'{data_folder}-eval-{feature}-eval-spoof.txt')

            gen_score_f = open(gen_score, 'r')

            for line in gen_score_f:
                gen_scores[feature].append(float(line))

            gen_score_f.close()

            df_score_f = open(df_score, 'r')

            for line in df_score_f:
                df_scores[feature].append(float(line))

            df_score_f.close()

        score_save_path = data_root.joinpath('cross-dataset').joinpath(f'{eval_dataset}-eval')
        os.makedirs(score_save_path, exist_ok=True)
        gen_score_f_name = score_save_path.joinpath(f'cross-dataset-eval-eval-genuine.txt')
        gen_score_f = open(gen_score_f_name, 'a')

        df_score_f_name = score_save_path.joinpath(f'cross-dataset-eval-eval-spoof.txt')
        df_score_f = open(df_score_f_name, 'a')

        for feature in features:

            for score in gen_scores[feature]:
                gen_score_f.write(f'{str(score)}\n')

            for score in df_scores[feature]:
                df_score_f.write(f'{str(score)}\n')

        gen_score_f.close()
        df_score_f.close()

        eer_save_path = score_save_path.joinpath('eer')
        os.makedirs(eer_save_path, exist_ok=True)

        eerinf_cmd = [
            'geteerinf',
            '-i',
            str(df_score_f_name),
            '-g',
            str(gen_score_f_name),
            '-e',
            f'Cross-dataset validation - {eval_dataset}-train',
            '-sp',
            str(eer_save_path)
        ]

        subprocess.call(eerinf_cmd)
