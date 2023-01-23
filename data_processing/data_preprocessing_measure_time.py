#
# Measure processing time for each feature
#
import os
from pathlib import Path
import time

import librosa
import numpy as np
from tqdm import tqdm


def process_recording(f_name, label):
    # Load file
    y, sr = librosa.load(f_name, sr=16000)

    rec_len = y.shape[0]

    if rec_len < 32000:
        print(f"Discarding {f_name}, {rec_len} < 32000")
        return

    start_time = time.time()
    # Mel-spectrogram
    mel_s = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256, n_fft=1024, hop_length=256)
    measured_time['mel'] += time.time() - start_time

    start_time = time.time()
    # STFT-spectrogram
    s = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    measured_time['stft'] += time.time() - start_time

    start_time = time.time()
    # CQT-spectorgram
    c = np.abs(librosa.cqt(y, sr=sr, hop_length=256))
    measured_time['cqt'] += time.time() - start_time

    start_time = time.time()
    # VQT-spectrogram
    v = np.abs(librosa.vqt(y, sr=sr, hop_length=256))
    measured_time['vqt'] += time.time() - start_time

    start_time = time.time()
    # IIRT-spectrogram
    i = np.abs(librosa.iirt(y, sr=sr, win_length=1024, hop_length=256))
    measured_time['iirt'] += time.time() - start_time

    start_time = time.time()
    # MFCC-spectrogram
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    measured_time['mfcc'] += time.time() - start_time

    start_time = time.time()
    # Chroma-spectrogram
    cro = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=1024, hop_length=256)
    measured_time['chroma'] += time.time() - start_time


features = ['mel', 'stft', 'cqt', 'vqt', 'iirt', 'mfcc', 'chroma']
measured_time = {}
dir_types = ['validation']
data_root = Path(f'/Users/antonfirc/Documents/Skola/PHD/Tools/Dataset/FoR/for-2seconds/')
rec_folders = ['real']
rec_count = 0

for feature in features:
    measured_time[feature] = 0.0

for dir_type in dir_types:
    dir_path = data_root.joinpath(dir_type)

    for type in rec_folders:
        type_path = dir_path.joinpath(type)

        label = 0 if type == 'fake' else 1

        recordings = []

        for rec in tqdm(os.listdir(type_path)):
            if '.wav' not in rec:
                continue

            # Load file
            f_name = type_path.joinpath(rec)
            process_recording(f_name, label)
            rec_count += 1

for feature in features:
    print(f"{feature}: total - {measured_time[feature]}: per sample - {measured_time[feature] / rec_count}")
