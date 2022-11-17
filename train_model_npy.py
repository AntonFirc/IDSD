import getopt
import sys
import tensorflow as tf
import datetime
from model_average import build_model
import numpy as np

# python3 -m tensorboard.main --logdir logs/fit

data_dir = ''
run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
resume_training = False
image_size = (256, 126)
load_model_path = ''
model_directory = './models'

try:
    opts, args = getopt.getopt(sys.argv[1:], "hri:n:l:m:")
except getopt.GetoptError:
    print('test.py -i <dataset_dir> -n <run_name> [ -r ]')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print('test.py -i <dataset_dir>')
        sys.exit()
    elif opt == '-i':
        data_dir = arg
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

model, batch_size = build_model(resume_training, "{0}/{1}".format(model_directory, load_model_path))

data_train = np.load('for_training_mel.npy', allow_pickle=True, fix_imports=True)
train_examples = data_train[:, 0].tolist()
train_labels = data_train[:, 1].tolist()

data_test = np.load('for_eval_mel.npy', allow_pickle=True, fix_imports=True)
test_examples = data_test[:, 0].tolist()
test_labels = data_test[:, 1].tolist()

train_ds = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

train_ds = train_ds.batch(8, drop_remainder=True)
train_ds = train_ds.shuffle(8)
test_ds = test_ds.batch(8, drop_remainder=True)
test_ds = test_ds.shuffle(8)

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + run_name
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

tf.config.run_functions_eagerly(True)

model.fit(train_ds, epochs=500,
          callbacks=[tensorboard_callback, early_stopping_callback],
          shuffle=True, use_multiprocessing=True, validation_data=test_ds)

model.save_weights("{0}/{1}.h5".format(model_directory, run_name))
model.evaluate(test_ds, verbose=2)
