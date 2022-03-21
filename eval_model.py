import getopt
import sys
from multiprocessing.pool import ThreadPool

import tensorflow as tf
import datetime
from tqdm import tqdm
import os
import numpy as np
from model_average import build_model

test_cases = ['mel', 'stft', 'cqt', 'vqt', 'iirt']


def load_image(image_path):
    input_img = tf.keras.preprocessing.image.load_img(
        image_path, grayscale=False, color_mode='grayscale', target_size=(640, 480),
        interpolation='nearest')
    input_arr = tf.keras.preprocessing.image.img_to_array(input_img)
    input_arr = input_arr.reshape(640, 480)

    if data_type == 'eval':
        x_eval.append(input_arr)
        y_eval.append(1 if is_real_data else 0)


data_dir = ''
now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
training_rounds = 0

x_train, x_test, x_eval = [], [], []
y_train, y_test, y_eval = [], [], []
data_type = 'eval'
is_real_data = True
model_suffix = '_max_full'

try:
    opts, args = getopt.getopt(sys.argv[1:], "hi:m:")
except getopt.GetoptError:
    print('test.py -i <dataset_path>')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print('test.py -i <dataset_path>')
        sys.exit()
    elif opt == '-i':
        data_dir = arg
    elif opt == '-m':
        model_suffix = arg

resume_training = True
thread_cnt = 96
real_image_data, fake_image_data = [], []

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

f = open("evaL_result{0}_{1}.txt".format(model_suffix, now), "w")
f.write("case: genuine accept ; deepfake reject ; deepfake accept ; genuine reject\n")

for case in test_cases:

    model_name = "./models/{0}{1}".format(case, model_suffix)
    model, batch_size = build_model(True, model_name)

    real_image_data = []
    data_eval_real = '{0}/{1}/real'.format(data_dir, case)
    data_eval_fake = '{0}/{1}/fake'.format(data_dir, case)

    is_real_data = True

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
                    fake_image_data[30000:]
                ),
                'Eval fake',
                len(fake_image_data[30000:]),
                unit="images"
            )
        )

    # normalize eval data
    x_eval, y_eval = np.array(x_eval), np.array(y_eval)
    x_eval = x_eval / 255.0

    tf.data.Dataset.from_tensor_slices(x_eval)

    y_pred = model.predict(x_eval)

    false_accepts = 0
    true_accepts = 0
    false_rejects = 0
    true_rejects = 0

    # 1 = real data
    # 0 = deepfake data

    for i in tqdm(range(len(y_eval))):
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

    f.write("{0}: {1} ; {2} ; {3} ; {4}\n".format(case, true_accepts, true_rejects, false_accepts, false_rejects))
    x_eval, y_eval, y_pred = [], [], []

f.write("total genuine/deepfake: {0} / {1}\n".format(len(real_image_data), len(fake_image_data)))
f.close()
