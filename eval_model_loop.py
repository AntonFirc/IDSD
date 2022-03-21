import getopt
import sys
from multiprocessing.pool import ThreadPool
import tensorflow as tf
import datetime
from tqdm import tqdm
import os
from model_average import build_model


def load_image(image_path):
    input_img = tf.keras.preprocessing.image.load_img(
        image_path, grayscale=False, color_mode='grayscale', target_size=(640, 480),
        interpolation='nearest')
    input_arr = tf.keras.preprocessing.image.img_to_array(input_img)
    input_arr = input_arr.reshape(640, 480)
    input_arr = input_arr / 255.0
    return tf.expand_dims(input_arr, axis=0)


def test_case(case):
    false_accepts = 0
    true_accepts = 0
    false_rejects = 0
    true_rejects = 0

    model_name = "./models/{0}{1}".format(case, model_suffix)
    model, batch_size = build_model(True, model_name)

    data_eval_real = '{0}/{1}/real'.format(data_dir, case)
    data_eval_fake = '{0}/{1}/fake'.format(data_dir, case)

    real_trials = 0

    for filename in tqdm(os.listdir(data_eval_real), desc="{0} real".format(case)):
        real_trials += 1
        img_data = load_image("{0}/{1}".format(data_eval_real, filename))
        predict = model(img_data, training=False)
        if predict > 0.5:
            true_accepts += 1
        else:
            false_rejects += 1

    fake_trials = 0

    for filename in tqdm(os.listdir(data_eval_fake), desc="{0} fake".format(case)):
        fake_trials += 1
        img_data = load_image("{0}/{1}".format(data_eval_fake, filename))
        predict = model(img_data, training=False)
        if predict < 0.5:
            true_rejects += 1
        else:
            false_accepts += 1

    f.write("{0}: {1} ; {2} ; {3} ; {4} ; {5} ; {6}\n".format(case, true_accepts, true_rejects, false_accepts,
                                                              false_rejects, real_trials, fake_trials))


test_cases = ['mel', 'stft', 'cqt', 'vqt', 'iirt']
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

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

f = open("evaL_result{0}_{1}.txt".format(model_suffix, now), "w")
f.write("case: genuine accept ; deepfake reject ; deepfake accept ; genuine reject ; real trials ; fake trials\n")

with ThreadPool(len(test_cases)) as pool:
    list(
        tqdm(
            pool.imap(
                test_case,
                test_cases
            ),
            'Eval model',
            len(test_cases),
            unit="cases"
        )
    )

f.close()
