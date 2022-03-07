cuDNN version - 7.6
CUDA version - 10.1

###Metacentrum modules

module add anaconda3-2019.10
module add ffmpeg
module add cuda-10.1
module add cudnn-7.6.4-cuda10.0

source activate IDSD

###Mel-spectrogram

Epoch 58/100
250/250 [==============================] - 12s 49ms/step - loss: 0.0993 - accuracy: 0.9660

Test1:
34/34 - 1s - loss: 0.1551 - accuracy: 0.9421

Test2:
34/34 - 1s - loss: 0.5919 - accuracy: 0.7721

Test3:
34/34 - 1s - loss: 0.3828 - accuracy: 0.8419

Eval:
34/34 - 1s - loss: 0.6814 - accuracy: 0.8667

python3 df_test.py -i for-rerec/mel -n mel_max_10 -r 0
python3 df_test.py -i for-rerec/stft -n stft_max_10 -r 0
python3 df_test.py -i for-rerec/cqt -n cqt_max_10 -r 0
python3 df_test.py -i for-rerec/vqt -n vqt_max_10 -r 0
python3 df_test.py -i for-rerec/iirt -n iirt_max_10 -r 0
python3 df_test.py -i for-rerec/chroma -n chroma_max_10 -r 0
python3 df_test.py -i for-rerec/mfcc -n mfcc_max_10 -r 0

python3 df_test.py -i for-rerec/mel -n mel_max_20 -r 0
python3 df_test.py -i for-rerec/stft -n stft_max_20 -r 0
python3 df_test.py -i for-rerec/cqt -n cqt_max_20 -r 0
python3 df_test.py -i for-rerec/vqt -n vqt_max_20 -r 0
python3 df_test.py -i for-rerec/iirt -n iirt_max_20 -r 0
python3 df_test.py -i for-rerec/chroma -n chroma_max_20 -r 0
python3 df_test.py -i for-rerec/mfcc -n mfcc_max_20 -r 0

python3 df_test.py -i for-rerec/mel -n mel_max_40 -r 0
python3 df_test.py -i for-rerec/stft -n stft_max_40 -r 0
python3 df_test.py -i for-rerec/cqt -n cqt_max_40 -r 0
python3 df_test.py -i for-rerec/vqt -n vqt_max_40 -r 0
python3 df_test.py -i for-rerec/iirt -n iirt_max_40 -r 0
python3 df_test.py -i for-rerec/chroma -n chroma_max_40 -r 0
python3 df_test.py -i for-rerec/mfcc -n mfcc_max_40 -r 0

python3 df_test.py -i for-2-sec/mel -n mel_avg_10 -r 10

python3 eval_model.py -i ./dataset/for-rerec -m _max_10
python3 eval_model.py -i ./dataset/for-rerec -m _max_20
python3 eval_model.py -i ./dataset/for-rerec -m _max_40

python3 train_dataset.py -i for-2-sec/mel -n mel_avg_tot