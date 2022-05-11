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


def apply_mp3_conversion(recording_name):
    recording_path = '{0}/{1}'.format(dataset_directory, recording_name)
    final_path = '{0}/mp3/{1}/{2}/{3}'.format(output_directory, dataset_type, data_type, recording_name)
    temp_name = hash(recording_name)
    # temp_path = 'compress/{0}/{1}.mp3'.format(output_directory, temp_name)
    temp_path = '{0}/mp3/{1}/{2}/{3}.mp3'.format(output_directory, dataset_type, data_type, temp_name)

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
    final_path = '{0}/reduce/{1}/{2}/{3}'.format(output_directory, dataset_type, data_type, recording_name)

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
    final_path = '{0}/volume/{1}/{2}/{3}'.format(output_directory, dataset_type, data_type, recording_name)

    sox_args = [
        'sox',
        '-v',
        '0.5',
        str(recording_path),
        str(final_path),
    ]
    s = subprocess.call(sox_args)


def apply_white_noise(recording_name):
    recording_path = '{0}/{1}'.format(dataset_directory, recording_name)
    final_path = '{0}/noise/{1}/{2}/{3}'.format(output_directory, dataset_type, data_type, recording_name)
    temp_name = hash(recording_name)
    temp_path = '{0}/noise/{1}/{2}/{3}.wav'.format(output_directory, dataset_type, data_type, temp_name)
    temp_path_vol = '{0}/noise/{1}/{2}/{3}_lownoise.wav'.format(output_directory, dataset_type, data_type, temp_name)

    # sox_args = [
    #     'sox',
    #     '-n',
    #     '-b',
    #     '16',
    #     str(temp_path),
    #     'synth',
    #     '2.0',
    #     'whitenoise'
    # ]

    sox_args = [
        'sox',
        str(recording_path),
        str(temp_path),
        'synth',
        'whitenoise'
    ]

    s = subprocess.call(sox_args)

    sox_args = [
        'sox',
        '-v',
        '0.02',
        str(temp_path),
        str(temp_path_vol),
    ]

    s = subprocess.call(sox_args)

    sox_args = [
        'sox',
        '-m',
        str(recording_path),
        str(temp_path_vol),
        str(final_path),
    ]
    s = subprocess.call(sox_args)

    rm_args = [
        'rm',
        str(temp_path),
    ]

    s = subprocess.call(rm_args)

    rm_args = [
        'rm',
        str(temp_path_vol),
    ]

    s = subprocess.call(rm_args)


def apply_street_noise(recording_name):
    recording_path = '{0}/{1}'.format(dataset_directory, recording_name)
    final_path = '{0}/street/{1}/{2}/{3}'.format(output_directory, dataset_type, data_type, recording_name)

    sox_args = [
        'sox',
        '-m',
        str(recording_path),
        'dataset/noise_street.wav',
        str(final_path),
    ]
    s = subprocess.call(sox_args)


def apply_bird_noise(recording_name):
    recording_path = '{0}/{1}'.format(dataset_directory, recording_name)
    final_path = '{0}/birds/{1}/{2}/{3}'.format(output_directory, dataset_type, data_type, recording_name)

    sox_args = [
        'sox',
        '-m',
        str(recording_path),
        'dataset/noise_birds.wav',
        str(final_path),
    ]
    s = subprocess.call(sox_args)


def apply_ogg_conversion(recording_name):
    recording_path = '{0}/{1}'.format(dataset_directory, recording_name)
    final_path = '{0}/ogg/{1}/{2}/{3}'.format(output_directory, dataset_type, data_type, recording_name)
    temp_name = hash(recording_name)
    temp_path = '{0}/ogg/{1}/{2}/{3}.ogg'.format(output_directory, dataset_type, data_type, temp_name)

    sox_args = [
        'sox',
        str(recording_path),
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


def apply_aac_conversion(recording_name):
    recording_path = '{0}/{1}'.format(dataset_directory, recording_name)
    final_path = '{0}/aac/{1}/{2}/{3}'.format(output_directory, dataset_type, data_type, recording_name)
    temp_name = hash(recording_name)
    temp_path = '{0}/aac/{1}/{2}/{3}.aac'.format(output_directory, dataset_type, data_type, temp_name)

    sox_args = [
        'ffmpeg',
        '-i',
        str(recording_path),
        str(temp_path),
    ]
    s = subprocess.call(sox_args)

    sox_args = [
        'ffmpeg',
        '-i',
        str(temp_path),
        str(final_path),
    ]
    s = subprocess.call(sox_args)

    clean_args = [
        'rm',
        str(temp_path),
    ]
    s = subprocess.call(clean_args)


def apply_wma_conversion(recording_name):
    recording_path = '{0}/{1}'.format(dataset_directory, recording_name)
    final_path = '{0}/wma/{1}/{2}/{3}'.format(output_directory, dataset_type, data_type, recording_name)
    temp_name = hash(recording_name)
    temp_path = '{0}/wma/{1}/{2}/{3}.wma'.format(output_directory, dataset_type, data_type, temp_name)

    sox_args = [
        'ffmpeg',
        '-i',
        str(recording_path),
        str(temp_path),
    ]
    s = subprocess.call(sox_args)

    sox_args = [
        'ffmpeg',
        '-i',
        str(temp_path),
        str(final_path),
    ]
    s = subprocess.call(sox_args)

    clean_args = [
        'rm',
        str(temp_path),
    ]
    s = subprocess.call(clean_args)


def apply_m4v_conversion(recording_name):
    recording_path = '{0}/{1}'.format(dataset_directory, recording_name)
    final_path = '{0}/m4v/{1}/{2}/{3}'.format(output_directory, dataset_type, data_type, recording_name)
    temp_name = hash(recording_name)
    temp_path = '{0}/m4v/{1}/{2}/{3}.m4v'.format(output_directory, dataset_type, data_type, temp_name)

    sox_args = [
        'ffmpeg',
        '-i',
        str(recording_path),
        str(temp_path),
    ]
    s = subprocess.call(sox_args)

    sox_args = [
        'ffmpeg',
        '-i',
        str(temp_path),
        str(final_path),
    ]
    s = subprocess.call(sox_args)

    clean_args = [
        'rm',
        str(temp_path),
    ]
    s = subprocess.call(clean_args)


def apply_bitrate_manipulation(recording_name):
    recording_path = '{0}/{1}'.format(dataset_directory, recording_name)
    final_path = '{0}/bitrate/{1}/{2}/{3}'.format(output_directory, dataset_type, data_type, recording_name)
    temp_name = hash(recording_name)
    temp_path = '{0}/bitrate/{1}/{2}/{3}.wav'.format(output_directory, dataset_type, data_type, temp_name)

    sox_args = [
        'sox',
        str(recording_path),
        '-b',
        '8',
        str(temp_path),
    ]
    s = subprocess.call(sox_args)

    sox_args = [
        'sox',
        str(temp_path),
        '-b',
        '16',
        str(final_path),
    ]
    s = subprocess.call(sox_args)

    clean_args = [
        'rm',
        str(temp_path),
    ]
    s = subprocess.call(clean_args)


def apply_combined_manipulation(recording_name):
    recording_path = '{0}/{1}'.format(dataset_directory, recording_name)
    final_path = '{0}/combined/{1}/{2}/{3}'.format(output_directory, dataset_type, data_type, recording_name)
    temp_name = hash(recording_name)
    temp_path = '{0}/combined/{1}/{2}/{3}.wav'.format(output_directory, dataset_type, data_type, temp_name)
    temp_path_vol = '{0}/combined/{1}/{2}/{3}_lownoise.wav'.format(output_directory, dataset_type, data_type, temp_name)

    sox_args = [
        'sox',
        str(recording_path),
        str(temp_path),
        'synth',
        'whitenoise'
    ]
    s = subprocess.call(sox_args)

    sox_args = [
        'sox',
        '-v',
        '0.01',
        str(temp_path),
        str(temp_path_vol),
    ]
    s = subprocess.call(sox_args)

    rm_args = [
        'rm',
        str(temp_path),
    ]
    s = subprocess.call(rm_args)

    sox_args = [
        'sox',
        '-m',
        str(recording_path),
        str(temp_path_vol),
        str(temp_path),
    ]
    s = subprocess.call(sox_args)

    sox_args = [
        'sox',
        '-v',
        '0.98',
        str(temp_path),
        str(final_path),
        'sinc',
        '7.99999k-5k'
    ]
    s = subprocess.call(sox_args)

    rm_args = [
        'rm',
        str(temp_path),
        str(temp_path_vol)
    ]
    s = subprocess.call(rm_args)


def apply_overdrive(recording_name):
    recording_path = '{0}/{1}'.format(dataset_directory, recording_name)
    final_path = '{0}/overdrive/{1}/{2}/{3}'.format(output_directory, dataset_type, data_type, recording_name)

    sox_args = [
        'sox',
        '-v'
        '0.92',
        str(recording_path),
        str(final_path),
        'overdrive'
    ]
    s = subprocess.call(sox_args)


def apply_flanger(recording_name):
    recording_path = '{0}/{1}'.format(dataset_directory, recording_name)
    final_path = '{0}/flanger/{1}/{2}/{3}'.format(output_directory, dataset_type, data_type, recording_name)

    sox_args = [
        'sox',
        str(recording_path),
        str(final_path),
        'flanger'
    ]
    s = subprocess.call(sox_args)


def apply_downsample(recording_name):
    recording_path = '{0}/{1}'.format(dataset_directory, recording_name)
    final_path = '{0}/downsample/{1}/{2}/{3}'.format(output_directory, dataset_type, data_type, recording_name)

    sox_args = [
        'sox',
        str(recording_path),
        str(final_path),
        'downsample'
    ]
    s = subprocess.call(sox_args)


def apply_reverb(recording_name):
    recording_path = '{0}/{1}'.format(dataset_directory, recording_name)
    final_path = '{0}/reverb/{1}/{2}/{3}'.format(output_directory, dataset_type, data_type, recording_name)

    sox_args = [
        'sox',
        str(recording_path),
        str(final_path),
        'reverb'
    ]
    s = subprocess.call(sox_args)


try:
    opts, args = getopt.getopt(sys.argv[1:], "hmrvngwbscavi:t:o:",
                               ['mp3', 'ogg', 'wma', 'aac', 'm4v', 'flanger', 'overdrive', 'downsample', 'reverb'])
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
    elif opt == '--mp3':
        mod_types.append('mp3')
    elif opt == '--ogg':
        mod_types.append('ogg')
    elif opt == '--wma':
        mod_types.append('wma')
    elif opt == '--aac':
        mod_types.append('aac')
    elif opt == '--m4v':
        mod_types.append('m4v')
    elif opt == '-r':
        mod_types.append('reduce')
    elif opt == '-v':
        mod_types.append('volume')
    elif opt == '-n':
        mod_types.append('noise')
    elif opt == '-b':
        mod_types.append('bitrate')
    elif opt == '-c':
        mod_types.append('combined')
    elif opt == '-s':
        mod_types.append('street')
        mod_types.append('birds')
    elif opt == '--overdrive':
        mod_types.append('overdrive')
    elif opt == '--flanger':
        mod_types.append('flanger')
    elif opt == '--downsample':
        mod_types.append('downsample')
    elif opt == '--reverb':
        mod_types.append('reverb')

for file in os.listdir(dataset_directory):
    if file.endswith('.wav'):
        file_buffer.append(file)

os.makedirs(output_directory, exist_ok=True)

dataset_type = dataset_directory.split('/')[-2]  # validation / training ...
data_type = dataset_directory.split('/')[-1]  # fake / real

for item in mod_types:
    if item == 'mp3':
        os.makedirs('{0}/mp3/{1}/{2}'.format(output_directory, dataset_type, data_type), exist_ok=True)
        func = apply_mp3_conversion
    if item == 'aac':
        os.makedirs('{0}/aac/{1}/{2}'.format(output_directory, dataset_type, data_type), exist_ok=True)
        func = apply_aac_conversion
    if item == 'ogg':
        os.makedirs('{0}/ogg/{1}/{2}'.format(output_directory, dataset_type, data_type), exist_ok=True)
        func = apply_ogg_conversion
    if item == 'wma':
        os.makedirs('{0}/wma/{1}/{2}'.format(output_directory, dataset_type, data_type), exist_ok=True)
        func = apply_wma_conversion
    if item == 'm4v':
        os.makedirs('{0}/m4v/{1}/{2}'.format(output_directory, dataset_type, data_type), exist_ok=True)
        func = apply_m4v_conversion
    if item == 'reduce':
        os.makedirs('{0}/reduce/{1}/{2}'.format(output_directory, dataset_type, data_type), exist_ok=True)
        func = apply_frequency_reduction
    if item == 'volume':
        os.makedirs('{0}/volume/{1}/{2}'.format(output_directory, dataset_type, data_type), exist_ok=True)
        func = apply_volume_reduction
    if item == 'bitrate':
        os.makedirs('{0}/bitrate/{1}/{2}'.format(output_directory, dataset_type, data_type), exist_ok=True)
        func = apply_bitrate_manipulation
    if item == 'combined':
        os.makedirs('{0}/combined/{1}/{2}'.format(output_directory, dataset_type, data_type), exist_ok=True)
        func = apply_combined_manipulation
    if item == 'noise':
        os.makedirs('{0}/noise/{1}/{2}'.format(output_directory, dataset_type, data_type), exist_ok=True)
        func = apply_white_noise
    if item == 'street':
        os.makedirs('{0}/street/{1}/{2}'.format(output_directory, dataset_type, data_type), exist_ok=True)
        func = apply_street_noise
    if item == 'birds':
        os.makedirs('{0}/birds/{1}/{2}'.format(output_directory, dataset_type, data_type), exist_ok=True)
        func = apply_bird_noise
    if item == 'overdrive':
        os.makedirs('{0}/overdrive/{1}/{2}'.format(output_directory, dataset_type, data_type), exist_ok=True)
        func = apply_overdrive
    if item == 'flanger':
        os.makedirs('{0}/flanger/{1}/{2}'.format(output_directory, dataset_type, data_type), exist_ok=True)
        func = apply_flanger
    if item == 'downsample':
        os.makedirs('{0}/downsample/{1}/{2}'.format(output_directory, dataset_type, data_type), exist_ok=True)
        func = apply_downsample
    if item == 'reverb':
        os.makedirs('{0}/reverb/{1}/{2}'.format(output_directory, dataset_type, data_type), exist_ok=True)
        func = apply_reverb

    with ThreadPool(thread_cnt) as pool:
        list(
            tqdm(
                pool.imap(
                    func,
                    file_buffer
                ),
                item,
                len(file_buffer),
                unit='recordings'
            )
        )
