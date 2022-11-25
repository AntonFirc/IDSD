#
# Preprocess ASVSpoof 2019 LA dataset
#
import math
import os
from multiprocessing.pool import ThreadPool
from pathlib import Path
from tqdm import tqdm
import librosa
import numpy as np

mels, stfts, cqts, vqts, iirts, mfccs, chromas = [], [], [], [], [], [], []


def parallel_wrapper(args):
    process_recording(*args)


def process_recording(f_name, label):
    # Load file
    y_a, sr = librosa.load(f_name, sr=16000)

    rec_len = y_a.shape[0]

    if rec_len < 32000:
        print(f"Discarding {f_name}, {rec_len} < 32000")
        return

    fragments = math.trunc(rec_len / 32000)

    for j in range(fragments):
        y = y_a[32000*j:32000*(j+1)]

        # Mel-spectrogram
        mel_s = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256, n_fft=1024, hop_length=256)

        mel_p = np.array([np.asarray(mel_s), label], dtype=object)
        mels.append(mel_p)

        # STFT-spectrogram
        s = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))

        stft_p = np.array([np.asarray(s), label], dtype=object)
        stfts.append(stft_p)

        # CQT-spectorgram
        c = np.abs(librosa.cqt(y, sr=sr, hop_length=256))

        cqt_p = np.array([np.asarray(c), label], dtype=object)
        cqts.append(cqt_p)

        # VQT-spectrogram
        v = np.abs(librosa.vqt(y, sr=sr, hop_length=256))

        vqt_p = np.array([np.asarray(v), label], dtype=object)
        vqts.append(vqt_p)

        # IIRT-spectrogram
        i = np.abs(librosa.iirt(y, sr=sr, win_length=1024, hop_length=256))

        iirt_p = np.array([np.asarray(i), label], dtype=object)
        iirts.append(iirt_p)

        # MFCC-spectrogram
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

        mfcc_p = np.array([np.asarray(mfcc), label], dtype=object)
        mfccs.append(mfcc_p)

        # Chroma-spectrogram
        cro = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=1024, hop_length=256)

        chroma_p = np.array([np.asarray(cro), label], dtype=object)
        chromas.append(chroma_p)


data_types = ['train']
data_root = Path('/storage/brno2/home/deemax/brno6/datasets/LA')

for data_type in data_types:
    protocol_path = data_root.joinpath('ASVspoof2019_LA_cm_protocols').joinpath(
        f'ASVspoof2019.LA.cm.{data_type}.trn.txt')
    rec_path = data_root.joinpath(f'ASVspoof2019_LA_{data_type}').joinpath('flac')

    proto = open(protocol_path, 'r')

    if not os.path.exists('processed_data'):
        os.makedirs('processed_data')

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

    mels_np = np.array(mels)
    np.save(f'processed_data/ASVSpoof_{data_type}_mel.npy', mels_np, allow_pickle=True, fix_imports=True)

    stfts_np = np.array(stfts)
    np.save(f'processed_data/ASVSpoof_{data_type}_stft.npy', stfts_np, allow_pickle=True, fix_imports=True)

    cqts_np = np.array(cqts)
    np.save(f'processed_data/ASVSpoof_{data_type}_cqt.npy', cqts_np, allow_pickle=True, fix_imports=True)

    vqts_np = np.array(vqts)
    np.save(f'processed_data/ASVSpoof_{data_type}_vqt.npy', vqts_np, allow_pickle=True, fix_imports=True)

    iirts_np = np.array(iirts)
    np.save(f'processed_data/ASVSpoof_{data_type}_iirt.npy', iirts_np, allow_pickle=True, fix_imports=True)

    mfccs_np = np.array(mfccs)
    np.save(f'processed_data/ASVSpoof_{data_type}_mfcc.npy', mfccs_np, allow_pickle=True, fix_imports=True)

    chromas_np = np.array(chromas)
    np.save(f'processed_data/ASVSpoof_{data_type}_chroma.npy', chromas_np, allow_pickle=True, fix_imports=True)
