#
# Preprocess FoR dataset
#
import os
from multiprocessing.pool import ThreadPool
from pathlib import Path
from tqdm import tqdm
from data_processor import *

dir_types = ['testing', 'validation', 'training']
data_root = Path(f'/Users/antonfirc/Documents/Skola/PHD/Tools/Dataset/FoR/for-rerecorded/')
rec_folders = ['real_fixed', 'fake_fixed']

if not os.path.exists('../processed_data'):
    os.makedirs('../processed_data')

for dir_type in dir_types:
    dir_path = data_root.joinpath(dir_type)

    for type in rec_folders:
        type_path = dir_path.joinpath(type)

        label = 0 if type == 'fake_fixed' else 1

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
                    f'FoR-Rerec - {dir_type} - {type}',
                    len(recordings),
                    unit='files'
                )
            )

    save_processed_data(f'for_rerec_{dir_type}')
