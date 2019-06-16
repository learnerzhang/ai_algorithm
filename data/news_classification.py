#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-06-04 14:27
# @Author  : zhangzhen
# @Site    : 
# @File    : news_classification.py
# @Software: PyCharm
import os
import codecs
import collections

file_dir = '/Users/zhangzhen/data/news_classification'

train_file = 'train.txt'
test_file = 'test.word'
dev_file = 'dev.txt'


def static():
    import collections
    import pprint
    train_static_count = collections.defaultdict(int)
    test_static_count = collections.defaultdict(int)

    with codecs.open(os.path.join(file_dir, train_file), encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            clazz, _ = line.split('\t')
            train_static_count[clazz] = train_static_count[clazz] + 1

    with codecs.open(os.path.join(file_dir, test_file), encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            clazz, _ = line.split('\t')
            test_static_count[clazz] = test_static_count[clazz] + 1

    pprint.pprint(train_static_count)
    pprint.pprint(test_static_count)


def read_news_corpus(clazz_num=0):
    train_corpus, train_labels = [], []
    test_corpus, test_labels = [], []
    dev_corpus, dev_labels = [], []

    clazz_count = collections.defaultdict(int)
    with codecs.open(os.path.join(file_dir, train_file), encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            clazz, token_str = line.split('\t')
            tokens = str(token_str).split(' ')
            if clazz_num > 0:
                if clazz_count[clazz] > clazz_num:
                    continue
                else:
                    clazz_count[clazz] = clazz_count[clazz] + 1
                    train_corpus.append(tokens)
                    train_labels.append(clazz)

            else:
                train_corpus.append(tokens)
                train_labels.append(clazz)

    with codecs.open(os.path.join(file_dir, test_file), encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            clazz, token_str = line.split('\t')
            tokens = str(token_str).split(' ')
            test_corpus.append(tokens)
            test_labels.append(clazz)

    with codecs.open(os.path.join(file_dir, dev_file), encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            clazz, token_str = line.split('\t')
            tokens = str(token_str).split(' ')
            dev_corpus.append(tokens)
            dev_labels.append(clazz)
    return train_corpus, train_labels, test_corpus, test_labels, dev_corpus, dev_labels


def create_vocabulary():
    labelSet = set()

    tmp_tokens = []
    with codecs.open(os.path.join(file_dir, train_file), encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            clazz, token_str = line.split('\t')
            tokens = str(token_str).split(' ')
            tmp_tokens.extend(tokens)
            labelSet.add(clazz)

    with codecs.open(os.path.join(file_dir, test_file), encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            clazz, token_str = line.split('\t')
            tokens = str(token_str).split(' ')
            tmp_tokens.extend(tokens)
            labelSet.add(clazz)

    with codecs.open(os.path.join(file_dir, dev_file), encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            clazz, token_str = line.split('\t')
            tokens = str(token_str).split(' ')
            tmp_tokens.extend(tokens)
            labelSet.add(clazz)

    vocabularySet = set(tmp_tokens)
    print("total word num:", len(vocabularySet), "total class num:", len(labelSet))

    vocab2int = {voc: idx + 1 for idx, voc in enumerate(vocabularySet)}
    int2vocab = {idx: voc for voc, idx in vocab2int.items()}
    vocab2int['PAD'] = 0
    int2vocab[0] = 'PAD'

    label2int = {label: idx for idx, label in enumerate(labelSet)}
    int2label = {idx: label for label, idx in label2int.items()}

    return vocab2int, int2vocab, label2int, int2label


if __name__ == '__main__':
    static()
    vocab2int, int2vocab, label2int, int2label = create_vocabulary()
    print(label2int, '\n', int2label)
    # train_corpus, train_labels, test_corpus, test_labels, dev_corpus, dev_labels = read_news_corpus()
    # print("train:", len(train_corpus), len(train_labels))
    # print("test:", len(test_corpus), len(test_labels))
    # print("dev:", len(dev_corpus), len(dev_labels))
