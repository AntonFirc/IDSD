#
# Preprocess WaveFake dataset
#
import os
from multiprocessing.pool import ThreadPool
from pathlib import Path
from tqdm import tqdm
from data_processor import *

data_root = Path('/storage/brno2/home/deemax/brno6/datasets/generated_audio')
rec_path = data_root.joinpath('wavs')

if not os.path.exists('../processed_data'):
    os.makedirs('../processed_data')

recordings = []

for fld in os.listdir(data_root):
    fld_path = data_root.joinpath(fld)

    for rec in os.listdir(fld_path):
        if not rec.endswith('.wav'):
            continue

        r_label = 0  # only deepfake audio

        f_name = fld_path.joinpath(rec)
        recordings.append((f_name, r_label))

with ThreadPool() as pool:
    list(
        tqdm(
            pool.imap(
                parallel_wrapper,
                recordings
            ),
            f'WaveFake',
            len(recordings),
            unit='files'
        )
    )


save_processed_data('WaveFake')
