from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model
from tcn import TCN
import tensorflow as tf


def build_model():
    batch_size, time_steps, input_dim = 16, 640, 480
    batch_input_shape = (batch_size, time_steps, input_dim)
    input_shape = Input(batch_shape=batch_input_shape)

    # setting dropout for TCN layers?
    tower_1 = TCN(batch_input_shape=batch_input_shape, kernel_size=3, return_sequences=True, padding='causal',
                  dropout_rate=0.2)(input_shape)
    tower_1 = tf.keras.layers.MaxPooling1D(pool_size=10, strides=5, padding='valid')(tower_1)

    tower_2 = TCN(batch_input_shape=batch_input_shape, kernel_size=5, return_sequences=True, padding='causal',
                  dropout_rate=0.2)(input_shape)
    tower_2 = tf.keras.layers.MaxPooling1D(pool_size=10, strides=5, padding='valid')(tower_2)

    tower_3 = TCN(batch_input_shape=batch_input_shape, kernel_size=7, return_sequences=True, padding='causal',
                  dropout_rate=0.2)(input_shape)
    tower_3 = tf.keras.layers.MaxPooling1D(pool_size=10, strides=5, padding='valid')(tower_3)

    concat = tf.keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)

    normalized = tf.keras.layers.BatchNormalization()(concat)

    dropped = tf.keras.layers.Dropout(.2)(normalized)

    out = tf.keras.layers.Flatten()(dropped)
    out = Dense(1, activation='sigmoid')(out)

    model = Model(inputs=[input_shape], outputs=[out])
    return model, batch_size
