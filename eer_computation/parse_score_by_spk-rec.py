from pathlib import Path

import numpy as np
import pandas as pd


features = ['mel', 'stft', 'cqt', 'vqt', 'iirt', 'chroma', 'mfcc']

for feature in features:
    score_f_names = [
        Path(f'/Users/antonfirc/Documents/Skola/PHD/Publikace/2022/ESORICS-Nine_simple_tricks/IDSD/score/FULL-FM-rapidminer/FULL-FM-cs-eval-{feature}-eval-genuine.txt'),
        Path(f'/Users/antonfirc/Documents/Skola/PHD/Publikace/2022/ESORICS-Nine_simple_tricks/IDSD/score/FULL-FM-rapidminer/FULL-FM-cs-eval-{feature}-eval-spoof.txt'),
        Path(f'/Users/antonfirc/Documents/Skola/PHD/Publikace/2022/ESORICS-Nine_simple_tricks/IDSD/score/FULL-FM-rapidminer/FULL-FM-en-eval-{feature}-eval-genuine.txt'),
        Path(f'/Users/antonfirc/Documents/Skola/PHD/Publikace/2022/ESORICS-Nine_simple_tricks/IDSD/score/FULL-FM-rapidminer/FULL-FM-en-eval-{feature}-eval-spoof.txt'),
    ]

    parsed_data = []

    for score_f_name in score_f_names:

        score_f = open(score_f_name)

        for line in score_f:
            pcs = line.split(',')
            score = float(pcs[-1])

            spk_inf = pcs[0]

            spk_pcs = spk_inf.split('/')

            rec_name = spk_pcs[-1]
            spk_name = spk_pcs[-2]
            rec_type = spk_pcs[-3]

            parsed_data.append([spk_name, rec_type, rec_name, score])

    arr = np.asarray(parsed_data)
    pd.DataFrame(arr).to_csv(f'/Users/antonfirc/Documents/Skola/PHD/Publikace/2022/ESORICS-Nine_simple_tricks/IDSD/RapidMiner/full/stft/dip-{feature}-full-eval.csv')
