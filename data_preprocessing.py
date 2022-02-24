import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


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
           
            # here dst has destination directory,
            # root[src_prefix:] gives us relative
            # path from source directory
            # and dirname has folder names
            dirpath = os.path.join(dst, root[src_prefix:], dirname)
             
            # making the path which we made by
            # joining all of the above three
            os.mkdir(dirpath)
            

def dirtree_preprocessing(cwd):
    create_dirtree_without_files(os.path.join(cwd, 'for-2seconds'), os.path.join(cwd, 'mel'))
    create_dirtree_without_files(os.path.join(cwd, 'for-2seconds'), os.path.join(cwd, 'stft'))
    create_dirtree_without_files(os.path.join(cwd, 'for-2seconds'), os.path.join(cwd, 'cqt'))
    create_dirtree_without_files(os.path.join(cwd, 'for-2seconds'), os.path.join(cwd, 'vqt'))
    create_dirtree_without_files(os.path.join(cwd, 'for-2seconds'), os.path.join(cwd, 'iirt'))
    create_dirtree_without_files(os.path.join(cwd, 'for-2seconds'), os.path.join(cwd, 'mfcc'))
    create_dirtree_without_files(os.path.join(cwd, 'for-2seconds'), os.path.join(cwd, 'chroma'))


def mel(y, sr, filename):
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256, n_fft=1024, hop_length=256)
    
    fig, ax = plt.subplots()
    mels_dB = librosa.power_to_db(mels, ref=np.max)
    img = librosa.display.specshow(mels_dB, hop_length=256, x_axis='time', y_axis='mel', sr=sr, fmax=sr, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')

    out = filename.replace('for-2seconds', 'mel')
    out = "png".join(out.rsplit("wav", 1))
    plt.savefig(out)


def stft(y, filename):
    s =  np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    
    fig, ax = plt.subplots()
    s_dB = librosa.amplitude_to_db(s, ref=np.max)
    img = librosa.display.specshow(s_dB, hop_length=256, x_axis='time', y_axis='log', sr=16000, ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title('Power spectrogram')
    
    out = filename.replace('for-2seconds', 'stft')
    out = "png".join(out.rsplit("wav", 1))
    plt.savefig(out)


def cqt (y, sr, filename):
    c = np.abs(librosa.cqt(y, sr=sr, hop_length=256))
    
    fig, ax = plt.subplots()
    c_dB = librosa.amplitude_to_db(c, ref=np.max)
    img = librosa.display.specshow(c_dB, hop_length=256, sr=sr, x_axis='time', y_axis='cqt_note', fmax=sr, ax=ax)
    ax.set_title('Constant-Q power spectrum')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    
    out = filename.replace('for-2seconds', 'cqt')
    out = "png".join(out.rsplit("wav", 1))
    plt.savefig(out)


def vqt(y, sr, filename):
    v = np.abs(librosa.vqt(y, sr=sr, hop_length=256))
    
    fig, ax = plt.subplots()
    v_dB = librosa.amplitude_to_db(v, ref=np.max)
    img = librosa.display.specshow(v_dB, hop_length=256, sr=sr, x_axis='time', y_axis='cqt_note', fmax=sr, ax=ax)
    ax.set_title('Variable-Q power spectrum')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    
    out = filename.replace('for-2seconds', 'vqt')
    out = "png".join(out.rsplit("wav", 1))
    plt.savefig(out)
    

def iirt(y, sr, filename):
    d = np.abs(librosa.iirt(y, sr=sr, win_length=1024, hop_length=256))
    
    fig, ax = plt.subplots()
    d_dB = librosa.amplitude_to_db(d, ref=np.max)
    img = librosa.display.specshow(d_dB, hop_length=256, sr=sr, x_axis='time', y_axis='cqt_hz', fmax=sr, ax=ax)
    ax.set_title('Semitone spectrogram (iirt)')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    
    out = filename.replace('for-2seconds', 'iirt')
    out = "png".join(out.rsplit("wav", 1))
    plt.savefig(out)


# da sa vygenerovat z melspectrogramu
def mfcc(y, sr, filename):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    
    fig, ax = plt.subplots()
    img = librosa.display.specshow(mfccs, x_axis='time', sr=sr, ax=ax)
    ax.set_title('MFCC')
    fig.colorbar(img, ax=ax)
    
    out = filename.replace('for-2seconds', 'mfcc')
    out = "png".join(out.rsplit("wav", 1))
    plt.savefig(out)
    

def chroma(y, sr, filename):
    chromas = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=1024, hop_length=256)
    
    fig, ax = plt.subplots()
    img = librosa.display.specshow(chromas, hop_length=256, x_axis='time', sr=sr, ax=ax)
    ax.set_title('Chroma')
    fig.colorbar(img, ax=ax)
    
    out = filename.replace('for-2seconds', 'chroma')
    out = "png".join(out.rsplit("wav", 1))
    plt.savefig(out)
    
    
def processing(dirname):
    for filename in os.listdir(dirname):
        if os.path.isdir(os.path.join(dirname, filename)):
            processing(os.path.join(dirname, filename))
        else:
            file = os.path.join(dirname, filename)
            y, sr = librosa.load(file, sr=16000)
            
            mel(y, sr, file)
            stft(y, file)
            cqt(y, sr, file)
            vqt(y, sr, file)
            iirt(y, sr, file)
            mfcc(y, sr, file)
            chroma(y, sr, file)
            

if __name__ == "__main__":
    cwd = os.getcwd()
    print("Creating directories for each spectrogram category...")

    try:
        dirtree_preprocessing(cwd)
    except FileExistsError:
        print("WARNING: One or more directories already exists.")

    print("Creating spectrograms for each .wav file...")
    processing(".\\for-2seconds")
