import getopt
import sys
import tensorflow as tf
import datetime
from model import build_model, eval_model
import numpy as np

# python3 -m tensorboard.main --logdir logs/fit

data_dir = ''
run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_directory = './models'
batch_size = 128
load_model_path = None
train_path = '/processed_data/old/ASVSpoof_dev_mel.npy'
test_path = '/processed_data/old/ASVSpoof_dev_mel.npy'
eval_path = '/processed_data/old/ASVSpoof_dev_mel.npy'

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

data_train = np.load(train_path, allow_pickle=True, fix_imports=True)
train_examples = data_train[:, 0].tolist()
train_labels = data_train[:, 1].tolist()

print("Train dataset balance:")
print(np.unique(train_labels, return_counts=True))

data_test = np.load(test_path, allow_pickle=True, fix_imports=True)
test_examples = data_test[:, 0].tolist()
test_labels = data_test[:, 1].tolist()

print("Test dataset balance:")
print(np.unique(test_labels, return_counts=True))

data_eval = np.load(eval_path, allow_pickle=True, fix_imports=True)
eval_examples = data_eval[:, 0].tolist()
eval_labels = data_eval[:, 1].tolist()

print("Eval dataset balance:")
print(np.unique(eval_labels, return_counts=True))


train_ds = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))
eval_ds = tf.data.Dataset.from_tensor_slices((eval_examples, eval_labels))

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
eval_ds = eval_ds.map(lambda x, y: (normalization_layer(x), y))

train_ds = train_ds.shuffle(len(train_examples))
train_ds = train_ds.batch(batch_size)
test_ds = test_ds.shuffle(len(test_examples))
test_ds = test_ds.batch(batch_size)
eval_ds = eval_ds.shuffle(len(eval_examples))
eval_ds = eval_ds.batch(batch_size)


AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + run_name
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

tf.config.run_functions_eagerly(True)

class_weight = {0: 1.,
                1: 9.}

model = build_model(False,
                    "{0}/{1}".format(model_directory, load_model_path),
                    batch_size=batch_size,
                    time_steps=train_examples[0].shape[0],
                    input_dim=train_examples[0].shape[1])

model.fit(train_ds, epochs=500,
          callbacks=[tensorboard_callback, early_stopping_callback], class_weight=class_weight,
          shuffle=True, use_multiprocessing=True, validation_data=test_ds)

model.save_weights("{0}/{1}.h5".format(model_directory, run_name))
model.evaluate(eval_ds, verbose=2)

eval_model(model, eval_path, run_name)
