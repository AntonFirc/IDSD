import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

data_root = Path('/storage/brno2/home/deemax/IDSD_new/processed_data')
# data_root = Path('/Users/antonfirc/Documents/Skola/PHD/Publikace/2022/ESORICS-Nine_simple_tricks/IDSD/processed_data')

features = ['mel', 'stft', 'cqt', 'vqt', 'iirt', 'chroma', 'mfcc']

split_pcs = 10

for feature in tqdm(features):
    wavefake = data_root.joinpath(f'WaveFake_train_{feature}.npy')
    # wavefake = data_root.joinpath(f'for_rerec_testing_chroma.npy')

    data_wf = np.load(str(wavefake), allow_pickle=True, fix_imports=True)

    data_gen_wf = data_wf[data_wf[:,1] == 1]
    data_df_wf = data_wf[data_wf[:,1] == 0]

    wf_gen_pcs = np.array_split(data_gen_wf, split_pcs)
    wf_df_pcs = np.array_split(data_df_wf, split_pcs)

    data_buffer = None

    save_path = str(data_root.joinpath('learning_curve'))
    os.makedirs(save_path, exist_ok=True)

    for i in range(split_pcs):
        if data_buffer is None:
            data_buffer = np.concatenate((wf_gen_pcs[i], wf_df_pcs[i]))
        else:
            data_add = np.concatenate((wf_gen_pcs[i], wf_df_pcs[i]))
            data_buffer = np.concatenate((data_buffer, data_add))

        np.save(f'{save_path}/WaveFake_train_{feature}_{i}.npy', data_buffer, allow_pickle=True, fix_imports=True)
