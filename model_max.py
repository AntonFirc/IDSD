from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model
from tcn import TCN
import tensorflow as tf


def build_model(resume_training, run_name):
    batch_size, time_steps, input_dim = None, 640, 480
    input_shape = Input(shape=(time_steps, input_dim))

    pool_size, strides = 2, 1

    # setting dropout for TCN layers?
    tower_1 = TCN(input_shape=(time_steps, input_dim), kernel_size=3, return_sequences=True, padding='causal',
                  dropout_rate=0.2)(input_shape)
    tower_1 = tf.keras.layers.MaxPooling1D(pool_size=pool_size, strides=strides, padding='valid')(tower_1)

    tower_2 = TCN(input_shape=(time_steps, input_dim), kernel_size=5, return_sequences=True, padding='causal',
                  dropout_rate=0.2)(input_shape)
    tower_2 = tf.keras.layers.MaxPooling1D(pool_size=pool_size, strides=strides, padding='valid')(tower_2)

    tower_3 = TCN(input_shape=(time_steps, input_dim), kernel_size=7, return_sequences=True, padding='causal',
                  dropout_rate=0.2)(input_shape)
    tower_3 = tf.keras.layers.MaxPooling1D(pool_size=pool_size, strides=strides, padding='valid')(tower_3)

    concat = tf.keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)

    normalized = tf.keras.layers.BatchNormalization()(concat)

    dropped = tf.keras.layers.Dropout(.2)(normalized)

    out = tf.keras.layers.Flatten()(dropped)
    out = Dense(1, activation='sigmoid')(out)

    model = Model(inputs=[input_shape], outputs=[out])

    if resume_training:
        print("Resuming training for model: {0}".format('{0}.h5'.format(run_name)))
        model.load_weights('models/{0}.h5'.format(run_name))

    loss_fce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    model.compile(optimizer='adam',
                  loss=loss_fce,
                  metrics=['accuracy'])

    model.summary()

    return model, batch_size
