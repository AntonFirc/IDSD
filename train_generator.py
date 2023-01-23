import getopt
import sys
from numpy import random

import tensorflow as tf
import datetime
from model import build_model
import numpy as np
from sklearn.utils import class_weight


def create_dataset_generator(inputs, labels):
    def argument_free_generator():
        for inp, label in zip(inputs, labels):
            yield inp / 255.0, label

    return argument_free_generator


data_dir = ''
run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_directory = './models'
batch_size = 128
load_model_path = None
train_path = 'processed_data/for_training_vqt.npy'
test_path = 'processed_data/for_testing_vqt.npy'
eval_path = 'processed_data/for_validation_vqt.npy'

# beware! not all options are used and do anything...
try:
    opts, args = getopt.getopt(sys.argv[1:], "hri:e:t:n:l:m:")
except getopt.GetoptError:
    print('train.py -i <data_path> -n <run_name> [ -r ]')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print('train.py -i <dataset_dir>')
        sys.exit()
    elif opt == '-i':
        train_path = arg
    elif opt == '-e':
        eval_path = arg
    elif opt == '-t':
        test_path = arg
    elif opt == '-n':
        run_name = arg
    elif opt == '-m':
        model_directory = arg
    elif opt == '-r':
        resume_training = True
    elif opt == '-l':
        load_model_path = arg

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

if not load_model_path:
    load_model_path = "{0}/{1}".format(model_directory, run_name)

train_examples = []
train_labels = []
data_train = None

train_pcs = train_path.split(',')
for train_set in train_pcs:
    data_load = np.load(train_set, allow_pickle=True, fix_imports=True)
    if data_train is None:
        data_train = data_load
    else:
        data_train = np.concatenate((data_train, data_load))

random.shuffle(data_train)
train_examples += data_train[:, 0].tolist()
train_labels += data_train[:, 1].tolist()

print("Train dataset balance:")
print(np.unique(train_labels, return_counts=True))

class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)

print("Balancing dataset with class weight:")
print(class_weights)

test_examples = []
test_labels = []

test_pcs = test_path.split(',')
for test_set in test_pcs:
    data_test = np.load(test_set, allow_pickle=True, fix_imports=True)
    test_examples = data_test[:, 0].tolist()
    test_labels = data_test[:, 1].tolist()

print("Test dataset balance:")
print(np.unique(test_labels, return_counts=True))

train_generator = create_dataset_generator(train_examples, train_labels)

train_ds = tf.data.Dataset.from_generator(train_generator, (tf.float32, tf.int32),
                                          ((train_examples[0].shape[0], train_examples[0].shape[1]), ()))
test_ds = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.shuffle(len(test_examples))
test_ds = test_ds.batch(batch_size)

train_ds = train_ds.repeat().batch(batch_size=batch_size)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + run_name
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5)

tf.config.run_functions_eagerly(True)

model = build_model(False,
                    "{0}/{1}".format(model_directory, load_model_path),
                    batch_size=batch_size,
                    time_steps=train_examples[0].shape[0],
                    input_dim=train_examples[0].shape[1])

model.fit(train_ds, epochs=500,
          callbacks=[tensorboard_callback, early_stopping_callback, reduce_lr_callback],
          class_weight={0: class_weights[0], 1: class_weights[1]},
          use_multiprocessing=True, validation_data=test_ds,
          steps_per_epoch=len(train_examples) // batch_size)

model.save_weights("{0}/{1}.h5".format(model_directory, run_name))
