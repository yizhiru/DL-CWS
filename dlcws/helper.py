import os
from collections import defaultdict
import codecs
from typing import List

PAD = "[PAD]"
UNK = "[UNK]"

OTHER_LABEL = 'O'


def parse_label_seqs_to_dict(label_seqs):
    """解析标注序列，得到类别词典"""
    labels = set()
    for label_seq in label_seqs:
        for label in label_seq:
            labels.add(label)
    label2idx = {PAD: 0}
    labels = sorted(list(labels))
    for l in labels:
        label2idx[l] = len(label2idx)
    return label2idx


def parse_char_seqs_to_dict(char_seqs, min_freq=1):
    """解析字符序列，得到字符词典"""
    char2freq_dict = defaultdict(int)
    for seq in char_seqs:
        for ch in seq:
            char2freq_dict[ch] += 1
    token2idx = {PAD: 0, UNK: 1}
    for ch, freq in char2freq_dict.items():
        if freq >= min_freq:
            token2idx[ch] = len(token2idx)
    return token2idx


def read_bert_vocab(bert_model_path):
    """读取bert词典"""
    dict_path = os.path.join(bert_model_path, 'vocab.txt')
    token2idx = {}
    with open(dict_path, 'r', encoding='utf-8') as f:
        tokens = f.read().splitlines()
    for word in tokens:
        token2idx[word] = len(token2idx)
    return token2idx


def load_data(file_path) -> (List[List[str]], List[List[str]]):
    """加载已标注文本数据"""
    chars_seqs, labels_seqs = [], []
    with codecs.open(file_path, 'r', 'utf8') as reader:
        for line in reader:
            line = line.strip()
            # empty string
            if not line:
                continue
            chars, labels = label_segmented_text(line)
            chars_seqs.append(chars)
            labels_seqs.append(labels)
    return chars_seqs, labels_seqs


def label_segmented_text(text: str) -> (List[str], List[str]):
    """将已分词文本标注成BIES标签"""
    parts = text.split(' ')
    chars, labels = [], []
    for p in parts:
        if p:
            cs = list(p)
            if len(cs) == 1:
                chars.extend(cs)
                labels.append('S')
            else:
                chars.extend(cs)
                ls = ['B', 'E']
                for i in range(1, len(cs) - 1):
                    ls.insert(i, 'I')
                labels.extend(ls)
    return chars, labels
