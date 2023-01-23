#
# Preprocess LJspeech dataset
#
import os
from multiprocessing.pool import ThreadPool
from pathlib import Path
from tqdm import tqdm
from data_processor import *

data_root = Path('/Users/antonfirc/Documents/Skola/PHD/Publikace/2022/ESORICS-Nine_simple_tricks/IDSD/dataset/LJSpeech-1.1')
rec_path = data_root.joinpath('wavs')

if not os.path.exists('../processed_data'):
    os.makedirs('../processed_data')

recordings = []

for rec in os.listdir(rec_path):
    if not rec.endswith('.wav'):
        continue

    r_label = 1  # only genuine audio

    f_name = rec_path.joinpath(rec)
    recordings.append((f_name, r_label))

with ThreadPool() as pool:
    list(
        tqdm(
            pool.imap(
                parallel_wrapper,
                recordings
            ),
            f'LJspeech',
            len(recordings),
            unit='files'
        )
    )


save_processed_data('LJspeech')
