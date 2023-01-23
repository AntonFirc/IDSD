#
# Preprocess FoR dataset
#
import math
import os
from multiprocessing.pool import ThreadPool
from pathlib import Path
import librosa
import numpy as np
from tqdm import tqdm

dir_type = 'validation'
data_root = Path(f'//Users/antonfirc/Documents/Skola/PHD/Publikace/2022/ESORICS-Nine_simple_tricks/IDSD/dataset/DP/CS_dp/')
rec_folders = ['Real', 'DF-1k', 'DF-5k']
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
        y = y_a[32000 * j:32000 * (j + 1)]

        # Mel-spectrogram
        mel_s = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256, n_fft=1024, hop_length=256)

        mel_p = np.array([f_name, np.asarray(mel_s), label], dtype=object)
        mels.append(mel_p)

        # STFT-spectrogram
        s = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))

        stft_p = np.array([f_name, np.asarray(s), label], dtype=object)
        stfts.append(stft_p)

        # CQT-spectorgram
        c = np.abs(librosa.cqt(y, sr=sr, hop_length=256))

        cqt_p = np.array([f_name, np.asarray(c), label], dtype=object)
        cqts.append(cqt_p)

        # VQT-spectrogram
        v = np.abs(librosa.vqt(y, sr=sr, hop_length=256))

        vqt_p = np.array([f_name, np.asarray(v), label], dtype=object)
        vqts.append(vqt_p)

        # IIRT-spectrogram
        i = np.abs(librosa.iirt(y, sr=sr, win_length=1024, hop_length=256))

        iirt_p = np.array([f_name, np.asarray(i), label], dtype=object)
        iirts.append(iirt_p)

        # MFCC-spectrogram
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

        mfcc_p = np.array([f_name, np.asarray(mfcc), label], dtype=object)
        mfccs.append(mfcc_p)

        # Chroma-spectrogram
        cro = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=1024, hop_length=256)

        chroma_p = np.array([f_name, np.asarray(cro), label], dtype=object)
        chromas.append(chroma_p)


if not os.path.exists('../processed_data'):
    os.makedirs('../processed_data')

for type in rec_folders:
    type_path = data_root.joinpath(type)

    label = 0 if 'DF' in type else 1

    recordings = []

    for spk in os.listdir(type_path):
        if ".DS_Store" in spk:
            continue

        spk_path = type_path.joinpath(spk)

        for rec in os.listdir(spk_path):
            if '.wav' not in rec:
                continue

            # Load file
            f_name = spk_path.joinpath(rec)
            recordings.append((f_name, label))

    with ThreadPool() as pool:
        list(
            tqdm(
                pool.imap(
                    parallel_wrapper,
                    recordings
                ),
                f'{dir_type} - {type}',
                len(recordings),
                unit='files'
            )
        )

mels_np = np.array(mels)
np.save(f'processed_data/dip_cs_{dir_type}_mel.npy', mels_np, allow_pickle=True, fix_imports=True)

stfts_np = np.array(stfts)
np.save(f'processed_data/dip_cs_{dir_type}_stft.npy', stfts_np, allow_pickle=True, fix_imports=True)

cqts_np = np.array(cqts)
np.save(f'processed_data/dip_cs_{dir_type}_cqt.npy', cqts_np, allow_pickle=True, fix_imports=True)

vqts_np = np.array(vqts)
np.save(f'processed_data/dip_cs_{dir_type}_vqt.npy', vqts_np, allow_pickle=True, fix_imports=True)

iirts_np = np.array(iirts)
np.save(f'processed_data/dip_cs_{dir_type}_iirt.npy', iirts_np, allow_pickle=True, fix_imports=True)

mfccs_np = np.array(mfccs)
np.save(f'processed_data/dip_cs_{dir_type}_mfcc.npy', mfccs_np, allow_pickle=True, fix_imports=True)

chromas_np = np.array(chromas)
np.save(f'processed_data/dip_cs_{dir_type}_chroma.npy', chromas_np, allow_pickle=True, fix_imports=True)

# geteerinf -p /Users/antonfirc/Documents/Skola/PHD/Publikace/2022/ESORICS-Nine_simple_tricks/IDSD/score/ASV2019_LA_eval -i ASV2019-eval-stft-eval-spoof.txt -g ASV2019-eval-stft-eval-genuine.txt -e "ASVSpoof 2019 eval STFT" -sp /Users/antonfirc/Documents/Skola/PHD/Publikace/2022/ESORICS-Nine_simple_tricks/IDSD/score/ASV2019_LA_eval/stft-ASV2019
# geteerinf -p /Users/antonfirc/Documents/Skola/PHD/Publikace/2022/ESORICS-Nine_simple_tricks/IDSD/score/FoR -i FoR-eval-stft-eval-spoof.txt -g FoR-eval-stft-eval-genuine.txt -e "FoR eval MFCC" -sp /Users/antonfirc/Documents/Skola/PHD/Publikace/2022/ESORICS-Nine_simple_tricks/IDSD/score/FoR/stft-FoR

# geteerinf -p /Users/antonfirc/Documents/Skola/PHD/Publikace/2022/ESORICS-Nine_simple_tricks/IDSD/score/dip -i cqt-spoof.txt -g cqt-genuine.txt -e "ASVSpoof model 2019 DIP eval CQT" -sp /Users/antonfirc/Documents/Skola/PHD/Publikace/2022/ESORICS-Nine_simple_tricks/IDSD/score/dip/dip_FoR2sec_CQT

