import getopt
import sys

import numpy as np
import tensorflow as tf
from model import build_model, eval_model

model_path = 'models/mel.h5'
run_name = 'for-2sec-mel'
data_path = 'for_eval_mel.npy'

try:
    opts, args = getopt.getopt(sys.argv[1:], "hri:n:l:m:")
except getopt.GetoptError:
    print('eval_model.py -i <data_path> -n <run_name> -m <model_path>')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print('train_model.py -i <dataset_dir>')
        sys.exit()
    elif opt == '-i':
        data_path = arg
    elif opt == '-n':
        run_name = arg
    elif opt == '-m':
        model_path = arg

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

data_eval = np.load(data_path, allow_pickle=True, fix_imports=True)
eval_examples = data_eval[:, 0].tolist()

model = build_model(True, model_path, time_steps=eval_examples[0].shape[0], input_dim=eval_examples[0].shape[1])

eval_model(model, data_path, run_name)
