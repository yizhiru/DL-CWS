import argparse
import os
import sys

import keras

sys.path.append('../')

from dlcws.model import CWS
from dlcws import helper

# 参数配置
parser = argparse.ArgumentParser()
parser.add_argument('-epochs', type=int, default=50)
parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-device_map', type=str, default='3')
parser.add_argument('-model_path', type=str, default='model')
parser.add_argument('-data_set', type=str, default='pku')
args = parser.parse_args()
param_str = '\n'.join(['%20s = %s' % (k, v) for k, v in sorted(vars(args).items())])
print('usage: %s\n%20s   %s\n%s\n%s\n' % (' '.join(sys.argv), 'ARG', 'VALUE', '_' * 50, param_str))

os.environ['CUDA_VISIBLE_DEVICES'] = args.device_map

root_path = '../data'

X_train, y_train = helper.load_data(os.path.join(root_path, args.data_set + '_training.utf8'))
X_valid, y_valid = helper.load_data(os.path.join(root_path, args.data_set + '_valid.utf8'))
X_test, y_test = helper.load_data(os.path.join(root_path, args.data_set + '_test.utf8'))

char2idx = helper.parse_char_seqs_to_dict(X_train + X_valid)

cws = CWS(char2idx, model_type='mlp')

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_crf_accuracy', patience=8),
    keras.callbacks.ModelCheckpoint(filepath=os.path.join(args.model_path, 'cws.h5'),
                                    monitor='val_crf_accuracy',
                                    save_best_only=True,
                                    save_weights_only=False)
]
cws.fit(X_train,
        y_train,
        X_val=X_valid,
        y_val=y_valid,
        epochs=args.epochs,
        batch_size=args.batch_size,
        fit_kwargs={'callbacks': callbacks})

# save word dict and label dict
cws.save_dict(args.model_path)

# load model
cws = CWS.load_model(os.path.join(args.model_path, 'ner.h5'),
                     dict_root_path=args.model_path)

y_pred = cws.predict(X_test, batch_size=args.batch_size)

result_path = os.path.join('../result', args.data_set + '_seg.txt')
with open(result_path, 'w', encoding='utf8') as fw:
    for words in y_pred:
        fw.write(' '.join(words) + '\n')
