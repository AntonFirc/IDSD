import os
from multiprocessing.pool import ThreadPool
import getopt
import librosa
import librosa.display
import matplotlib.pyplot
import numpy as np
from tqdm import tqdm
import sys


def create_dirtree_without_files(src, dst):
    # getting the absolute path of the source
    # directory
    src = os.path.abspath(src)

    # making a variable having the index till which
    # src string has directory and a path separator
    src_prefix = len(src) + len(os.path.sep)

    # making the destination directory
    os.makedirs(dst)

    # doing os walk in source directory
    for root, dirs, files in os.walk(src):
        for dirname in dirs:
            if dirname == 'flac':
                continue
            # here dst has destination directory,
            # root[src_prefix:] gives us relative
            # path from source directory
            # and dirname has folder names
            dirpath = os.path.join(dst, root[src_prefix:], dirname)

            # making the path which we made by
            # joining all of the above three
            os.mkdir(dirpath)


def dirtree_preprocessing(src_path, dst_path):
    create_dirtree_without_files(os.path.join(src_path), os.path.join(dst_path, 'mel'))
    create_dirtree_without_files(os.path.join(src_path), os.path.join(dst_path, 'stft'))
    create_dirtree_without_files(os.path.join(src_path), os.path.join(dst_path, 'cqt'))
    create_dirtree_without_files(os.path.join(src_path), os.path.join(dst_path, 'vqt'))
    create_dirtree_without_files(os.path.join(src_path), os.path.join(dst_path, 'iirt'))
    create_dirtree_without_files(os.path.join(src_path), os.path.join(dst_path, 'mfcc'))
    create_dirtree_without_files(os.path.join(src_path), os.path.join(dst_path, 'chroma'))


def mel(y, sr, filename):
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256, n_fft=1024, hop_length=256)

    plt = matplotlib.pyplot
    fig, ax = plt.subplots()
    mels_dB = librosa.power_to_db(mels, ref=np.max)
    img = librosa.display.specshow(mels_dB, hop_length=256, x_axis='time', y_axis='mel', sr=sr, fmax=sr, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')

    out = output_path + '/mel/{0}/{1}/{2}'.format(filename.split('/')[-3], filename.split('/')[-2],
                                                  filename.split('/')[-1])
    out = "png".join(out.rsplit("wav", 1))
    plt.savefig(out)
    plt.close()


def stft(y, filename):
    s = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))

    plt = matplotlib.pyplot
    fig, ax = plt.subplots()
    s_dB = librosa.amplitude_to_db(s, ref=np.max)
    img = librosa.display.specshow(s_dB, hop_length=256, x_axis='time', y_axis='log', sr=16000, ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title('Power spectrogram')

    out = output_path + '/stft/{0}/{1}/{2}'.format(filename.split('/')[-3], filename.split('/')[-2],
                                                  filename.split('/')[-1])
    out = "png".join(out.rsplit("wav", 1))
    plt.savefig(out)
    plt.close()


def cqt(y, sr, filename):
    c = np.abs(librosa.cqt(y, sr=sr, hop_length=256))

    plt = matplotlib.pyplot
    fig, ax = plt.subplots()
    c_dB = librosa.amplitude_to_db(c, ref=np.max)
    img = librosa.display.specshow(c_dB, hop_length=256, sr=sr, x_axis='time', y_axis='cqt_note', fmax=sr, ax=ax)
    ax.set_title('Constant-Q power spectrum')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

    out = output_path + '/cqt/{0}/{1}/{2}'.format(filename.split('/')[-3], filename.split('/')[-2],
                                                  filename.split('/')[-1])
    out = "png".join(out.rsplit("wav", 1))
    plt.savefig(out)
    plt.close()


def vqt(y, sr, filename):
    v = np.abs(librosa.vqt(y, sr=sr, hop_length=256))

    plt = matplotlib.pyplot
    fig, ax = plt.subplots()
    v_dB = librosa.amplitude_to_db(v, ref=np.max)
    img = librosa.display.specshow(v_dB, hop_length=256, sr=sr, x_axis='time', y_axis='cqt_note', fmax=sr, ax=ax)
    ax.set_title('Variable-Q power spectrum')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

    out = output_path + '/vqt/{0}/{1}/{2}'.format(filename.split('/')[-3], filename.split('/')[-2],
                                                  filename.split('/')[-1])
    out = "png".join(out.rsplit("wav", 1))
    plt.savefig(out)
    plt.close()


def iirt(y, sr, filename):
    d = np.abs(librosa.iirt(y, sr=sr, win_length=1024, hop_length=256))

    plt = matplotlib.pyplot
    fig, ax = plt.subplots()
    d_dB = librosa.amplitude_to_db(d, ref=np.max)
    img = librosa.display.specshow(d_dB, hop_length=256, sr=sr, x_axis='time', y_axis='cqt_hz', fmax=sr, ax=ax)
    ax.set_title('Semitone spectrogram (iirt)')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

    out = output_path + '/iirt/{0}/{1}/{2}'.format(filename.split('/')[-3], filename.split('/')[-2],
                                                  filename.split('/')[-1])
    out = "png".join(out.rsplit("wav", 1))
    plt.savefig(out)
    plt.close()


# da sa vygenerovat z melspectrogramu
def mfcc(y, sr, filename):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    plt = matplotlib.pyplot
    fig, ax = plt.subplots()
    img = librosa.display.specshow(mfccs, x_axis='time', sr=sr, ax=ax)
    ax.set_title('MFCC')
    fig.colorbar(img, ax=ax)

    out = output_path + '/mfcc/{0}/{1}'.format(filename.split('/')[-2], filename.split('/')[-1])
    out = "png".join(out.rsplit("wav", 1))
    plt.savefig(out)
    plt.close()


def chroma(y, sr, filename):
    chromas = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=1024, hop_length=256)

    plt = matplotlib.pyplot
    fig, ax = plt.subplots()
    img = librosa.display.specshow(chromas, hop_length=256, x_axis='time', sr=sr, ax=ax)
    ax.set_title('Chroma')
    fig.colorbar(img, ax=ax)

    out = output_path + '/chroma/{0}/{1}'.format(filename.split('/')[-2], filename.split('/')[-1])
    out = "png".join(out.rsplit("wav", 1))
    plt.savefig(out)
    plt.close()


def process_image(file):
    y, sr = librosa.load(file, sr=16000)

    mel(y, sr, file)
    stft(y, file)
    cqt(y, sr, file)
    vqt(y, sr, file)
    iirt(y, sr, file)
    # mfcc(y, sr, file)
    # chroma(y, sr, file)


def processing(dirname):
    input_files = []
    for filename in os.listdir(dirname):
        if filename == '.DS_Store' or filename == 'flac':
            continue
        if os.path.isdir(os.path.join(dirname, filename)):
            processing(os.path.join(dirname, filename))
        else:
            if filename.__contains__('.wav'):
                file = os.path.join(dirname, filename)
                input_files.append(file)

    if input_files:
        input_files = input_files
        with ThreadPool(thread_cnt) as pool:
            list(
                tqdm(
                    pool.imap(
                        process_image,
                        input_files
                    ),
                    dirname,
                    len(input_files),
                    unit='files'
                )
            )


thread_cnt = 1
project_path = '/storage/brno6/home/deemax/IDSD/dataset'
dataset_name = 'for-2-sec_compress'
output_dir = 'for-2-sec_compress'

try:
    opts, args = getopt.getopt(sys.argv[1:], "hp:d:o:")
except getopt.GetoptError:
    print('modify_recordings.py -p <project_path> -d <dataset_name> -o <output_dir>')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print('modify_recordings.py -i <dataset_dir> -t <thread_cnt>')
        sys.exit()
    elif opt == '-p':
        project_path = arg
    elif opt == '-d':
        dataset_name = arg
    elif opt == '-o':
        output_dir = arg

if __name__ == "__main__":
    cwd = os.getcwd()
    # dataset_path = cwd + "/dataset/{0}".format(dataset_name)
    dataset_path = "{0}/{1}".format(project_path, dataset_name)
    # print("Creating directories for each spectrogram category...")
    # output_path = cwd + "/dataset/{0}".format(output_dir)
    output_path = "{0}/{1}".format(project_path, output_dir)

    # try:
    #     dirtree_preprocessing(dataset_path, output_path)
    # except FileExistsError:
    #     print("WARNING: One or more directories already exists.")

    print("Creating spectrograms for each .wav file...")
    processing(dataset_path)
