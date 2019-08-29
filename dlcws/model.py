import json
import os
import random
from typing import List, Dict

import keras
from keras import backend as K
from keras.layers import Dense
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_losses
from keras_contrib.metrics import crf_accuracies
from keras_preprocessing import sequence

from dlcws import helper as H


class CWS:

    def __init__(self,
                 token2idx: Dict = None,
                 embedding_dim=100,
                 model_type='mlp',
                 lr=0.001):
        self.token2idx = token2idx
        self.label2idx = {H.PAD: 0, 'B': 1, 'I': 2, 'E': 3, 'S': 4}
        self.idx2label = {i: l for l, i in self.label2idx.items()}
        self.embedding_dim = embedding_dim
        self.model_type = model_type
        self.lr = lr
        self.model: keras.models.Model = None

    def __tokenize(self, sentences: List[List[str]]) -> List[List[int]]:
        """字符映射index"""
        return [[self.token2idx.get(token, self.token2idx[H.UNK]) for token in sentence]
                for sentence in sentences]

    def __convert_label_seqs_to_idx(self, label_seqs: List[List[str]]) -> List[List[int]]:
        """label映射index"""
        return [[self.label2idx[l] for l in label_seq]
                for label_seq in label_seqs]

    def __convert_idx_seq(self, chars, indices, raw_length):
        """将类别id序列转成分词结果"""
        labels = [self.idx2label[idx] for idx in indices[: raw_length]]
        words = []
        i = 0
        while i < raw_length:
            l = labels[i]
            if l == 'S':
                words.append(chars[i])
            elif l == 'B':
                # 尾字符或下一次字符不能连续成词
                if i == raw_length - 1 or labels[i + 1] not in {'I', 'E'}:
                    words.append(chars[i])
                else:
                    w = chars[i]
                    for j in range(i + 1, raw_length):
                        if chars[j] == 'I':
                            w += chars[j]
                        elif chars[j] == 'E':
                            w += chars[j]
                            i = j + 1
                            break
                        else:
                            i = j
                            break
            else:
                words.append(chars[i])
        return words

    def __convert_idx_seqs(self,
                           chars_seqs: List[List[str]],
                           idx_seqs: List[List[int]],
                           raw_lens: List[int]) -> List[List[str]]:
        """将众多类别ID序列转成分词结果"""
        seg_list = []
        for chars, idx_seq, raw_len in zip(chars_seqs, idx_seqs, raw_lens):
            seg_list.append(self.__convert_idx_seq(chars, idx_seq, raw_len))
        return seg_list

    def __data_generator(self,
                         x: List[List[str]],
                         y: List[List[str]],
                         sequence_len: int,
                         batch_size: int):
        while True:
            steps = (len(x) + batch_size - 1) // batch_size
            # shuffle data
            xy = list(zip(x, y))
            random.shuffle(xy)
            x, y = zip(*xy)
            for i in range(steps):
                batch_x = x[i * batch_size: (i + 1) * batch_size]
                batch_y = y[i * batch_size: (i + 1) * batch_size]

                tokenized_x = self.__tokenize(batch_x)
                idx_y = self.__convert_label_seqs_to_idx(batch_y)

                padded_x = sequence.pad_sequences(tokenized_x,
                                                  maxlen=sequence_len,
                                                  padding='post',
                                                  truncating='post',
                                                  value=self.token2idx[H.PAD])
                padded_y = sequence.pad_sequences(idx_y,
                                                  maxlen=sequence_len,
                                                  padding='post',
                                                  truncating='post',
                                                  value=self.label2idx[H.PAD])
                one_hot_y = keras.utils.to_categorical(padded_y, num_classes=len(self.label2idx))
                yield (padded_x, one_hot_y)

    def __build_model(self):
        input_layer = keras.layers.Input(shape=(None,))
        embedding_layer = keras.layers.Embedding(input_dim=len(self.token2idx),
                                                 output_dim=self.embedding_dim,
                                                 mask_zero=True,
                                                 trainable=True,
                                                 name='Embedding')(input_layer)
        if self.model_type == 'bi-lstm':
            hidden_layer = keras.layers.Bidirectional(keras.layers.LSTM(units=256,
                                                                        recurrent_dropout=0.4,
                                                                        return_sequences=True),
                                                      name='Bi-LSTM')(embedding_layer)
        elif self.model_type == 'gru':
            hidden_layer = keras.layers.GRU(units=256,
                                            recurrent_dropout=0.4,
                                            return_sequences=True,
                                            name='GRU')(embedding_layer)
        else:
            hidden_layer = Dense(units=512, activation=K.sigmoid, name='dense')(embedding_layer)

        dense_layer = keras.layers.TimeDistributed(Dense(units=256,
                                                         activation=K.relu),
                                                   name='td_dense')(hidden_layer)
        crf_layer = CRF(units=len(self.label2idx), sparse_target=False, name='CRF')(dense_layer)
        model = keras.models.Model(inputs=input_layer, outputs=crf_layer)
        model.compile(optimizer=keras.optimizers.Adam(lr=self.lr),
                      loss=crf_losses.crf_loss,
                      metrics=[crf_accuracies.crf_accuracy])
        model.summary()
        self.model = model

    def fit(self,
            X_train,
            y_train,
            X_val,
            y_val,
            batch_size=64,
            epochs=50,
            fit_kwargs: Dict = None):
        if self.model is None:
            self.__build_model()
        # 最长序列长度
        max_seq_len = max([len(x) for x in X_train + X_val])

        if len(X_train) < batch_size:
            batch_size = len(X_train) // 2

        train_generator = self.__data_generator(X_train, y_train, max_seq_len, batch_size)

        if fit_kwargs is None:
            fit_kwargs = {}

        if X_val:
            val_generator = self.__data_generator(X_val, y_val, max_seq_len, batch_size)
            fit_kwargs['validation_data'] = val_generator
            fit_kwargs['validation_steps'] = (len(X_val) + batch_size - 1) // batch_size

        self.model.fit_generator(train_generator,
                                 steps_per_epoch=(len(X_train) + batch_size - 1) // batch_size,
                                 epochs=epochs,
                                 **fit_kwargs)

    def predict(self,
                sentences: List[List[str]],
                batch_size=64) -> List[List[str]]:
        tokens = self.__tokenize(sentences)
        raw_len_seqs = [len(sentence) for sentence in sentences]
        max_seq_len = max([len(x) for x in sentences])
        padded_tokens = sequence.pad_sequences(tokens,
                                               maxlen=max_seq_len,
                                               padding='post',
                                               truncating='post',
                                               value=self.token2idx[H.PAD])
        pred_prob_seqs = self.model.predict(padded_tokens, batch_size=batch_size)
        idx_seqs = pred_prob_seqs.argmax(-1)

        return self.__convert_idx_seqs(sentences, idx_seqs, raw_len_seqs)

    @classmethod
    def get_custom_objects(cls):
        return {'CRF': CRF,
                'crf_loss': crf_losses.crf_loss,
                'crf_accuracy': crf_accuracies.crf_accuracy}

    def save_dict(self, dict_root_path):
        with open(os.path.join(dict_root_path, 'vocab.json'), 'w', encoding='utf8') as fw:
            fw.write(json.dumps(self.token2idx, indent=2, ensure_ascii=False))
        with open(os.path.join(dict_root_path, 'labels.json'), 'w', encoding='utf8') as fw:
            fw.write(json.dumps(self.label2idx, indent=2, ensure_ascii=False))

    @classmethod
    def load_model(cls, model_path, dict_root_path):
        agent = cls()
        agent.model = keras.models.load_model(model_path, custom_objects=cls.get_custom_objects())
        agent.model.summary()
        with open(os.path.join(dict_root_path, 'vocab.json'), 'r', encoding='utf8') as fr:
            agent.token2idx = json.load(fr)
        with open(os.path.join(dict_root_path, 'labels.json'), 'r', encoding='utf8') as fr:
            agent.label2idx = json.load(fr)
        agent.idx2label = {v: k for k, v in agent.label2idx.items()}
        return agent
