import os
import subprocess
from pathlib import Path

data_root = Path('/score')
data_folder = 'WF-FM'

score_root = data_root.joinpath(data_folder)

features = ['mel', 'stft', 'cqt', 'vqt', 'iirt', 'mfcc', 'chroma']

for feature in features:
    gen_score_cs = score_root.joinpath(f'{data_folder}-cs-eval-{feature}-eval-genuine.txt')
    df_score_cs = score_root.joinpath(f'{data_folder}-cs-eval-{feature}-eval-spoof.txt')

    gen_score_en = score_root.joinpath(f'{data_folder}-en-eval-{feature}-eval-genuine.txt')
    df_score_en = score_root.joinpath(f'{data_folder}-en-eval-{feature}-eval-spoof.txt')

    eer_save_path = score_root.joinpath(feature)
    os.makedirs(eer_save_path, exist_ok=True)

    eerinf_cmd = [
        'geteerinf',
        '-i',
        f"{df_score_cs},{df_score_en}",
        '-g',
        f"{gen_score_cs},{gen_score_en}",
        '-e',
        data_folder,
        '-sp',
        str(eer_save_path)
    ]

    subprocess.call(eerinf_cmd)
