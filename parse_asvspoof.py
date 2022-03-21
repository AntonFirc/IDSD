import os
from multiprocessing.pool import ThreadPool
from pathlib import Path
import subprocess
import audioread
from tqdm import tqdm

thread_cnt = 96
dataset_path = Path('dataset/ASVspoof2021_DF_eval')
flac_dir_path = dataset_path.joinpath('flac')
key_file_name = 'trial_metadata.txt'
key_file_path = dataset_path.joinpath(key_file_name)
fake_data_path = dataset_path.joinpath('fake')
real_data_path = dataset_path.joinpath('real')


def process_recording(recording):
    split = recording.split('-')
    recording_file = split[1].strip() + '.flac'
    recording_file_path = flac_dir_path.joinpath(recording_file)
    target_file_name = recording_file.replace('.flac', '.wav')

    recording_class = split[-2].strip()
    final_path = real_data_path.joinpath(target_file_name)
    if recording_class == 'spoof':
        final_path = fake_data_path.joinpath(target_file_name)

    with audioread.audio_open(str(recording_file_path)) as f:
        if f.duration < 2:
            return

    convert_args = [
        'sox',
        '--norm',
        str(recording_file_path),
        str(final_path),
        'silence',
        '1',
        '0.1',
        '1%',
        'trim',
        '0',
        '00:02'
    ]

    s = subprocess.call(convert_args)


key_file = open(key_file_path, 'r')
lines = key_file.read().splitlines()  # puts the file into an array.
key_file.close()

os.mkdir(fake_data_path)
os.mkdir(real_data_path)

with ThreadPool(thread_cnt) as pool:
    list(
        tqdm(
            pool.imap(
                process_recording,
                lines
            ),
            'Parse ASVspoof',
            len(lines),
            unit='recordings'
        )
    )
