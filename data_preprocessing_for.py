import os
from pathlib import Path
import librosa
import numpy as np
from tqdm import tqdm

data_root = Path('/Users/antonfirc/Documents/Skola/PHD/Tools/Dataset/FoR/for-2seconds/validation')
rec_folders = ['real', 'fake']

specs = []

for type in rec_folders:
    type_path = data_root.joinpath(type)

    label = 0 if type == 'fake' else 1

    for rec in tqdm(os.listdir(type_path)):
        if '.wav' not in rec:
            continue

        f_name = type_path.joinpath(rec)
        y, sr = librosa.load(f_name, sr=16000)
        mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256, n_fft=1024, hop_length=256)

        pair = np.array([np.asarray(mels), label], dtype=object)

        specs.append(pair)

specs = np.array(specs)
np.save('for_eval_mel.npy', specs, allow_pickle=True, fix_imports=True)
