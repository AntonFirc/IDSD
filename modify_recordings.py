import getopt
import os
import subprocess
import sys
from multiprocessing.pool import ThreadPool
from tqdm import tqdm

dataset_directory = ''
output_directory = ''
mod_types = []
file_buffer = []
thread_cnt = 4


def apply_compression(recording_name):
    recording_path = '{0}/{1}'.format(dataset_directory, recording_name)
    final_path = '{0}/compress/{1}'.format(output_directory, recording_name)
    temp_name = hash(recording_name)
    temp_path = '{0}/compress/{1}.mp3'.format(output_directory, temp_name)

    sox_args = [
        'sox',
        str(recording_path),
        '-C',
        '9',
        str(temp_path),
    ]
    s = subprocess.call(sox_args)

    sox_args = [
        'sox',
        str(temp_path),
        str(final_path),
    ]
    s = subprocess.call(sox_args)

    clean_args = [
        'rm',
        str(temp_path),
    ]
    s = subprocess.call(clean_args)


def apply_frequency_reduction(recording_name):
    recording_path = '{0}/{1}'.format(dataset_directory, recording_name)
    final_path = '{0}/reduce/{1}'.format(output_directory, recording_name)

    sox_args = [
        'sox',
        str(recording_path),
        str(final_path),
        'sinc',
        '7.99999k-5k'
    ]
    s = subprocess.call(sox_args)


def apply_volume_reduction(recording_name):
    recording_path = '{0}/{1}'.format(dataset_directory, recording_name)
    final_path = '{0}/volume/{1}'.format(output_directory, recording_name)

    sox_args = [
        'sox',
        '-v',
        '0.5',
        str(recording_path),
        str(final_path),
    ]
    s = subprocess.call(sox_args)


try:
    opts, args = getopt.getopt(sys.argv[1:], "hcrvi:t:o:")
except getopt.GetoptError:
    print('modify_recordings.py -i <dataset_dir>')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print('modify_recordings.py -i <dataset_dir> -t <thread_cnt>')
        sys.exit()
    elif opt == '-i':
        dataset_directory = arg
    elif opt == '-o':
        output_directory = arg
    elif opt == '-t':
        thread_cnt = int(arg)
    elif opt == '-c':
        mod_types.append('compress')
    elif opt == '-r':
        mod_types.append('reduce')
    elif opt == '-v':
        mod_types.append('volume')

for file in os.listdir(dataset_directory):
    if file.endswith('.wav'):
        file_buffer.append(file)

os.makedirs(output_directory, exist_ok=True)

for item in mod_types:
    if item == 'compress':
        os.makedirs('{0}/compress'.format(output_directory), exist_ok=True)
        func = apply_compression
    if item == 'reduce':
        os.makedirs('{0}/reduce'.format(output_directory), exist_ok=True)
        func = apply_frequency_reduction
    if item == 'volume':
        os.makedirs('{0}/volume'.format(output_directory), exist_ok=True)
        func = apply_volume_reduction

    with ThreadPool(thread_cnt) as pool:
        list(
            tqdm(
                pool.imap(
                    func,
                    file_buffer
                ),
                dataset_directory,
                len(file_buffer),
                unit='recordings'
            )
        )
