from pathlib import Path

import numpy as np
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model
from tcn import TCN
import tensorflow as tf


def build_model(resume_training, model_path, batch_size=None):
    time_steps, input_dim = 256, 126
    input_shape_tuple = (batch_size, time_steps, input_dim)
    input_shape = Input(batch_input_shape=input_shape_tuple)

    pool_size, strides = 10, 5

    # setting dropout for TCN layers?
    tower_1 = TCN(input_shape=input_shape_tuple, kernel_size=3, return_sequences=True, padding='causal',
                  dropout_rate=0.2)(input_shape)
    tower_1 = tf.keras.layers.AveragePooling1D(pool_size=pool_size, strides=strides, padding='valid')(tower_1)

    tower_2 = TCN(input_shape=input_shape_tuple, kernel_size=5, return_sequences=True, padding='causal',
                  dropout_rate=0.2)(input_shape)
    tower_2 = tf.keras.layers.AveragePooling1D(pool_size=pool_size, strides=strides, padding='valid')(tower_2)

    tower_3 = TCN(input_shape=input_shape_tuple, kernel_size=7, return_sequences=True, padding='causal',
                  dropout_rate=0.2)(input_shape)
    tower_3 = tf.keras.layers.AveragePooling1D(pool_size=pool_size, strides=strides, padding='valid')(tower_3)

    concat = tf.keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)

    normalized = tf.keras.layers.BatchNormalization()(concat)

    dropped = tf.keras.layers.Dropout(.2)(normalized)

    out = tf.keras.layers.Flatten()(dropped)
    out = Dense(1, activation='sigmoid')(out)

    model = Model(inputs=[input_shape], outputs=[out])

    if resume_training:
        print("Resuming training for model: {0}".format('{0}.h5'.format(model_path)))
        model.load_weights('{0}.h5'.format(model_path))

    loss_fce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    model.compile(optimizer='adam',
                  loss=loss_fce,
                  metrics=['accuracy'])

    model.summary()

    return model


def eval_model(model, eval_data_path):
    data_eval = np.load(eval_data_path, allow_pickle=True, fix_imports=True)
    x_eval = np.array(data_eval[:, 0].tolist())
    y_eval = data_eval[:, 1].tolist()

    x_eval = x_eval / 255.0

    preds = model.predict(x_eval)

    gen_score_f_name = Path('score/for-2sec-mel-eval-genuine.txt')
    df_score_f_name = Path('score/for-2sec-mel-eval-spoof.txt')

    gen_score_f = open(gen_score_f_name, 'w')
    df_score_f = open(df_score_f_name, 'w')

    for i in range(len(preds)):
        if y_eval[i] == 0:
            df_score_f.write("%1.4f\n" % preds[i][0])
        if y_eval[i] == 1:
            gen_score_f.write("%1.4f\n" % preds[i][0])

    gen_score_f.close()
    df_score_f.close()
