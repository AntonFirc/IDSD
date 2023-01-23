#
# Preprocess AVCeleb dataset
#
import os
from multiprocessing.pool import ThreadPool
from pathlib import Path
from tqdm import tqdm
from data_processor import *

dir_type = 'full'
data_root = Path('/Users/antonfirc/Documents/Skola/PHD/Publikace/2022/ESORICS-Nine_simple_tricks/IDSD/dataset/AVCeleb')
rec_folders = ['real', 'fake']
mels, stfts, cqts, vqts, iirts, mfccs, chromas = [], [], [], [], [], [], []

if not os.path.exists('../processed_data'):
    os.makedirs('../processed_data')

for type in rec_folders:
    type_path = data_root.joinpath(type)

    label = 0 if type == 'fake' else 1

    recordings = []

    for rec in os.listdir(type_path):
        if '.wav' not in rec:
            continue

        # Load file
        f_name = type_path.joinpath(rec)
        recordings.append((f_name, label))

    with ThreadPool() as pool:
        list(
            tqdm(
                pool.imap(
                    parallel_wrapper,
                    recordings
                ),
                f'AVCeleb - {dir_type} - {type}',
                len(recordings),
                unit='files'
            )
        )

save_processed_data(f'AVCeleb_{dir_type}')
