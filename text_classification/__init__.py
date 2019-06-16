#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-06-04 14:55
# @Author  : zhangzhen
# @Site    : 
# @File    : __init__.py
# @Software: PyCharm
from typing import List
import numpy as np


def generate_ngram_tokens(tokens: List, ngram=2):
    if len(tokens) >= ngram:
        ngram_tokens = ["".join(tokens[i - ngram: i]) for i in range(ngram, len(tokens) + 1)]
        return ngram_tokens + tokens
    return tokens


def corpus_filter(train_corpus, train_labels, test_corpus, test_labels, dev_corpus, dev_labels,
                  labels=None, ngram=1):
    """
    过滤选取目标数据集
    :param train_corpus:
    :param train_labels:
    :param test_corpus:
    :param test_labels:
    :param dev_corpus:
    :param dev_labels:
    :param labels:
    :return:
    """
    if labels:
        train_corpus = [dat for dat, label in zip(train_corpus, train_labels) if label in labels]
        train_labels = [label for label in train_labels if label in labels]

        test_corpus = [dat for dat, label in zip(test_corpus, test_labels) if label in labels]
        test_labels = [label for label in test_labels if label in labels]

        dev_corpus = [dat for dat, label in zip(dev_corpus, dev_labels) if label in labels]
        dev_labels = [label for label in dev_labels if label in labels]
    # generate n-gram

    if ngram > 1:
        ngram = int(ngram)
        train_corpus = [generate_ngram_tokens(dat, ngram) for dat in train_corpus]
        test_corpus = [generate_ngram_tokens(dat, ngram) for dat in test_corpus]
        dev_corpus = [generate_ngram_tokens(dat, ngram) for dat in dev_corpus]

    return train_corpus, train_labels, test_corpus, test_labels, dev_corpus, dev_labels


def data2Vecter(corpus, labels, vocab2int, label2int, sequence_length=30, sparse=True):
    X = [[vocab2int[token] for token in tokens] + (sequence_length - len(tokens)) * [0] for tokens in corpus]
    if sparse:
        y = [label2int[label] for label in labels]
    else:
        y = []
        for label in labels:
            tmp = np.zeros(len(label2int))
            tmp[label2int[label]] = 1
            y.append(tmp)

    assert len(X) == len(y)
    return X, y


def batch(X, y, batch_size=8, shuttle=True, seed=1234):
    total_num = len(X)
    if shuttle:
        indices = list(range(total_num))
        np.random.seed(seed=seed)
        np.random.shuffle(indices)
        X = [X[idx] for idx in indices]
        y = [y[idx] for idx in indices]

    batch_num = int((total_num + batch_size - 1) / batch_size)
    for i in range(batch_num):
        if i + 1 * batch_size > total_num:
            yield (X[i * batch_size:], y[i * batch_size:])
        else:
            yield (X[i * batch_size: (i + 1) * batch_size], y[i * batch_size: (i + 1) * batch_size])


if __name__ == '__main__':
    tokens = ['A', 'B', 'C', 'D']
    tokens = generate_ngram_tokens(tokens=tokens, ngram=2)
    print(tokens)

    # ======# ======# ======# ======# ======
    X = np.random.randint(0, 100, size=(9, 10))
    y = np.random.rand(9)
    print(X, y)
    for (b_x, b_y) in batch(X, y):
        print(b_x[0], b_y[0])
