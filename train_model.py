import getopt
import sys
import tensorflow as tf
import datetime
from model_average import build_model

# python3 -m tensorboard.main --logdir logs/fit

data_dir = ''
run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
resume_training = False
image_size = (640, 480)
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

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    color_mode='grayscale',
    seed=123,
    image_size=image_size)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    subset="validation",
    color_mode='grayscale',
    validation_split=0.2,
    seed=123,
    image_size=image_size)

class_names = train_ds.class_names
print(class_names)

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + run_name
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

model.fit(train_ds, epochs=500,
          callbacks=[tensorboard_callback, early_stopping_callback],
          shuffle=True, use_multiprocessing=True, validation_data=test_ds)

model.save_weights("{0}/{1}.h5".format(model_directory, run_name))
model.evaluate(test_ds, verbose=2)
