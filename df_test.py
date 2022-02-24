import tensorflow as tf
from keras.layers import Dense
from keras.models import Input, Model
from keras.utils.vis_utils import plot_model
from tcn import TCN
import datetime

# python3 -m tensorboard.main --logdir logs/fit

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#TODO - change!
num_classes = 10

# TODO - remove after providing valid dataset
mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
# remove end

batch_size, time_steps, input_dim = 32, 28, 28
batch_input_shape = (batch_size, time_steps, input_dim)
input_shape = Input(batch_shape=batch_input_shape)

# setting dropout for TCN layers?
tower_1 = TCN(batch_input_shape=batch_input_shape, kernel_size=3, return_sequences=True, padding='causal')(input_shape)
tower_1 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='valid')(tower_1)

tower_2 = TCN(batch_input_shape=batch_input_shape, kernel_size=5, return_sequences=True, padding='causal')(input_shape)
tower_2 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='valid')(tower_2)

tower_3 = TCN(input_shape=(time_steps, input_dim), kernel_size=7, return_sequences=True, padding='causal')(input_shape)
tower_3 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='valid')(tower_3)

concat = tf.keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)

normalized = tf.keras.layers.BatchNormalization()(concat)

dropped = tf.keras.layers.Dropout(.2)(normalized)

out = tf.keras.layers.Flatten()(dropped)
out = Dense(num_classes, activation='sigmoid')(out)

model = Model(inputs=[input_shape], outputs=[out])

# plot_model(model, to_file='network_image.png')
# predictions = model(x_train[:1]).numpy()

loss_fce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])

model.summary()

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

model.fit(x_train, y_train, epochs=50, callbacks=[tensorboard_callback, early_stopping_callback])

#model.evaluate(x_test, y_test, verbose=2)

# https://github.com/philipperemy/keras-tcn
# https://stackoverflow.com/questions/43151775/how-to-have-parallel-convolutional-layers-in-keras
