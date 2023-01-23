import getopt
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from model import build_model, eval_model
import subprocess

feature_names = ['mel', 'stft', 'vqt', 'cqt', 'iirt', 'mfcc', 'chroma']


print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

data_path_arg = 'processed_data/dip_en_validation_'
run_name_arg = 'FULL-FM-en-eval-'
model_path_arg = 'models/Full-'
save_path_arg = 'score/FULL-FM'

try:
    opts, args = getopt.getopt(sys.argv[1:], "hd:n:m:s:")
except getopt.GetoptError:
    print('eval_model.py getopt error')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print('eval_models.py')
        sys.exit()
    elif opt == '-d':
        data_path_arg = arg
    elif opt == '-n':
        run_name_arg = arg
    elif opt == '-m':
        model_path_arg = arg
    elif opt == '-s':
        save_path_arg = arg

for feature_name in feature_names:

    # model_path = 'models/mel.h5'
    model_path = f'{model_path_arg}{feature_name}.h5'
    run_name = f'{run_name_arg}{feature_name}'
    # data_path = f'/storage/brno2/home/deemax/IDSD_new/processed_data/for_validation_{feature_name}.npy'
    data_path = f'{data_path_arg}{feature_name}.npy'
    save_path = Path(save_path_arg)

    data_eval = np.load(data_path, allow_pickle=True, fix_imports=True)
    x_eval = np.array(data_eval[:, 1].tolist())
    x_eval = x_eval / 255.0
    y_eval = data_eval[:, 2].tolist()

    model = build_model(True, model_path, time_steps=x_eval[0].shape[0], input_dim=x_eval[0].shape[1])

    x_eval_pcs = np.array_split(x_eval, 5)

    preds = []

    for pcs in x_eval_pcs:
        prediction = model.predict(pcs)
        preds = preds + prediction.tolist()

    gen_score_f_name = save_path.joinpath(f'{run_name}-eval-genuine.txt')
    df_score_f_name = save_path.joinpath(f'{run_name}-eval-spoof.txt')

    gen_score_f = open(gen_score_f_name, 'w')
    df_score_f = open(df_score_f_name, 'w')

    for i in range(len(preds)):
        if y_eval[i] == 0:
            df_score_f.write("%1.4f\n" % preds[i][0])
        if y_eval[i] == 1:
            gen_score_f.write("%1.4f\n" % preds[i][0])

    gen_score_f.close()
    df_score_f.close()

    eer_save_path = save_path.joinpath(feature_name)
    os.makedirs(eer_save_path, exist_ok=True)

    eerinf_cmd = [
        'geteerinf',
        '-i',
        df_score_f_name,
        '-g',
        gen_score_f_name,
        '-e',
        run_name,
        '-sp',
        str(eer_save_path)
    ]

    subprocess.call(eerinf_cmd)