import random
from multiprocessing.pool import ThreadPool
import tensorflow as tf
import datetime
from tqdm import tqdm
import os
import numpy as np
from model import build_model


# python3 -m tensorboard.main --logdir logs/fit

def load_image(image_path):
    input_img = tf.keras.preprocessing.image.load_img(
        image_path, grayscale=False, color_mode='grayscale', target_size=(640, 480),
        interpolation='nearest')
    input_arr = tf.keras.preprocessing.image.img_to_array(input_img)
    input_arr = input_arr.reshape(640, 480)

    if data_type == 'train':
        x_train.append(input_arr)
        y_train.append(1 if is_real_data else 0)
    if data_type == 'test':
        x_test.append(input_arr)
        y_test.append(1 if is_real_data else 0)
    if data_type == 'eval':
        x_eval.append(input_arr)
        y_eval.append(1 if is_real_data else 0)


resume_training = True
thread_cnt = 8

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

data_path_real = './dataset/mel_upl/mel/training/real'
data_path_fake = './dataset/mel_upl/mel/training/fake'
step_size = 2000

x_train, x_test, x_eval = [], [], []
y_train, y_test, y_eval = [], [], []

data_test_real = './dataset/mel_upl/mel/testing/real'
data_test_fake = './dataset/mel_upl/mel/testing/fake'

data_type = 'test'
is_real_data = True

real_image_data = []

for filename in os.listdir(data_test_real):
    real_image_data.append("{0}/{1}".format(data_test_real, filename))

with ThreadPool(thread_cnt) as pool:
    list(
        tqdm(
            pool.imap(
                load_image,
                real_image_data
            ),
            'Test real',
            len(real_image_data),
            unit="images"
        )
    )

fake_image_data = []
is_real_data = False

for filename in os.listdir(data_test_fake):
    fake_image_data.append("{0}/{1}".format(data_test_fake, filename))

with ThreadPool(thread_cnt) as pool:
    list(
        tqdm(
            pool.imap(
                load_image,
                fake_image_data
            ),
            'Test fake',
            len(fake_image_data),
            unit="images"
        )
    )

data_eval_real = './dataset/mel_upl/mel/validation/real'
data_eval_fake = './dataset/mel_upl/mel/validation/fake'

data_type = 'eval'
is_real_data = True

real_image_data = []

for filename in os.listdir(data_eval_real):
    real_image_data.append("{0}/{1}".format(data_eval_real, filename))

with ThreadPool(thread_cnt) as pool:
    list(
        tqdm(
            pool.imap(
                load_image,
                real_image_data
            ),
            'Eval real',
            len(real_image_data),
            unit="images"
        )
    )

fake_image_data = []
is_real_data = False

for filename in os.listdir(data_eval_fake):
    fake_image_data.append("{0}/{1}".format(data_eval_fake, filename))

with ThreadPool(thread_cnt) as pool:
    list(
        tqdm(
            pool.imap(
                load_image,
                fake_image_data
            ),
            'Eval fake',
            len(fake_image_data),
            unit="images"
        )
    )

# normalize test data
x_test, y_test = np.array(x_test), np.array(y_test)
x_test = x_test / 255.0

# normalize eval data
x_eval, y_eval = np.array(x_eval), np.array(y_eval)
x_eval = x_eval / 255.0

model, batch_size = build_model()

if resume_training:
    print("Resuming training for model: {0}".format('mel_random_chkpt.h5'))
    model.load_weights('mel_random_chkpt.h5')

loss_fce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

model.compile(optimizer='adam',
              loss=loss_fce,
              metrics=['accuracy'])

model.summary()

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)


for i in range(0):
    print("Train loop: {}".format(i))
    data_type = 'train'
    is_real_data = True

    real_image_data = []
    cnt = 0
    idx_list = np.arange(6978)
    selected_samples_indices = random.sample(list(idx_list), 2000)

    for filename in os.listdir(data_path_real):
        if cnt in selected_samples_indices:
            real_image_data.append("{0}/{1}".format(data_path_real, filename))
        cnt += 1

    with ThreadPool(thread_cnt) as pool:
        list(
            tqdm(
                pool.imap(
                    load_image,
                    real_image_data
                ),
                'Train real',
                len(real_image_data),
                unit="images"
            )
        )

    fake_image_data = []
    is_real_data = False
    cnt = 0
    idx_list = np.arange(6978)
    selected_samples_indices = random.sample(list(idx_list), 2000)

    for filename in os.listdir(data_path_fake):
        if cnt in selected_samples_indices:
            fake_image_data.append("{0}/{1}".format(data_path_fake, filename))
        cnt += 1

    with ThreadPool(thread_cnt) as pool:
        list(
            tqdm(
                pool.imap(
                    load_image,
                    fake_image_data
                ),
                'Train fake',
                len(fake_image_data),
                unit="images"
            )
        )

    x_train_s1, y_train_s1 = np.array(x_train), np.array(y_train)
    x_train_s1 = x_train_s1 / 255.0
    x_train, y_train = [], []

    model.fit(x_train_s1, y_train_s1, epochs=100,
              callbacks=[tensorboard_callback, early_stopping_callback],
              shuffle=True, use_multiprocessing=True)

    model.save_weights("mel_random_chkpt.h5")
    model.evaluate(x_test, y_test, verbose=2)

print("Final evaluation:")
#model.evaluate(x_eval, y_eval, verbose=2)

y_pred = model.predict(x_eval)

false_accepts = 0
true_accepts = 0

false_rejects = 0
true_rejects = 0

# 1 = real data
# 0 = deepfake data

for i in range(len(y_eval)):
    # if deepfake
    if y_eval[i] == 0:
        if y_pred[i] < 0.5:
            true_rejects += 1
        else:
            false_accepts += 1
    # if real
    if y_eval[i] == 1:
        if y_pred[i] > 0.5:
            true_accepts += 1
        else:
            false_rejects += 1

eval_true = np.sum(y_eval)
eval_false = len(y_eval) - eval_true

print("Success rate: {0}%".format((true_accepts + true_rejects) / len(y_eval) * 100))
print("False accepts: {0}/{1}".format(false_accepts, eval_false))
print("True accepts: {0}/{1}".format(true_accepts, eval_true))
print("False rejects: {0}/{1}".format(false_rejects, eval_true))
print("True rejects: {0}/{1}".format(true_rejects, eval_false))

# https://github.com/philipperemy/keras-tcn
# https://stackoverflow.com/questions/43151775/how-to-have-parallel-convolutional-layers-in-keras
