import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

data_root = Path('/storage/brno2/home/deemax/IDSD_new/processed_data')

features = ['mel', 'stft', 'cqt', 'vqt', 'iirt', 'chroma', 'mfcc']

for feature in tqdm(features):
    jsut = data_root.joinpath(f'JSUT_{feature}.npy')
    ljspeech = data_root.joinpath(f'LJspeech_{feature}.npy')
    wavefake = data_root.joinpath(f'WaveFake_{feature}.npy')

    data_jsut = np.load(str(jsut), allow_pickle=True, fix_imports=True)
    data_lj = np.load(str(ljspeech), allow_pickle=True, fix_imports=True)
    data_wf = np.load(str(wavefake), allow_pickle=True, fix_imports=True)


    train_ratio = 0.7
    validation_ratio = 0.2
    test_ratio = 0.1

    jsut_train, jsut_test = train_test_split(data_jsut, test_size=1 - train_ratio)
    jsut_val, jsut_test = train_test_split(jsut_test,
                                           test_size=test_ratio / (test_ratio + validation_ratio))

    lj_train, lj_test = train_test_split(data_lj, test_size=1 - train_ratio)
    lj_val, lj_test = train_test_split(lj_test,
                                       test_size=test_ratio / (test_ratio + validation_ratio))

    wf_train, wf_test = train_test_split(data_wf, test_size=1 - train_ratio)
    wf_val, wf_test = train_test_split(wf_test,
                                       test_size=test_ratio / (test_ratio + validation_ratio))

    train = np.concatenate((jsut_train, lj_train, wf_train))
    dev = np.concatenate((jsut_test, lj_test, wf_test))
    eval = np.concatenate((jsut_val, lj_val, wf_val))

    np.save(f'../processed_data/WaveFake_train_{feature}.npy', train, allow_pickle=True, fix_imports=True)
    np.save(f'../processed_data/WaveFake_dev_{feature}.npy', dev, allow_pickle=True, fix_imports=True)
    np.save(f'../processed_data/WaveFake_eval_{feature}.npy', eval, allow_pickle=True, fix_imports=True)
