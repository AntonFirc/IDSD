#
# Preprocess ASVSpoof 2019 LA dataset
#
import os
from multiprocessing.pool import ThreadPool
from pathlib import Path
from tqdm import tqdm

from data_processor import *

data_types = ['eval']
data_root = Path(f'/Users/antonfirc/Documents/Skola/PHD/Publikace/2022/ESORICS-Nine_simple_tricks/IDSD/dataset/LA')
# data_root = Path('/storage/brno2/home/deemax/brno6/datasets/LA')

for data_type in data_types:
    protocol_path = data_root.joinpath('ASVspoof2019_LA_cm_protocols').joinpath(
        f'ASVspoof2019.LA.cm.{data_type}.trl.txt')
    rec_path = data_root.joinpath(f'ASVspoof2019_LA_{data_type}').joinpath('flac')

    proto = open(protocol_path, 'r')

    if not os.path.exists('../processed_data'):
        os.makedirs('../processed_data')

    recordings = []

    for line in proto:
        pcs = line.split()
        r_name = pcs[1] + '.flac'
        r_label = 0 if pcs[-1] == 'spoof' else 1

        f_name = rec_path.joinpath(r_name)
        recordings.append((f_name, r_label))

    with ThreadPool() as pool:
        list(
            tqdm(
                pool.imap(
                    parallel_wrapper,
                    recordings
                ),
                f'ASVSpoof - {data_type}',
                len(recordings),
                unit='files'
            )
        )

    save_processed_data(f'ASVSpoof_{data_type}')
