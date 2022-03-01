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