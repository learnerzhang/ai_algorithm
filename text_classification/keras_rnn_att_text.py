#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-06-06 15:00
# @Author  : zhangzhen
# @Site    : 
# @File    : rnn_text.py
# @Software: PyCharm
import logging
import argparse
from keras import Sequential, Input, Model
from keras.engine import Layer
from keras import backend as K
from keras.layers import Embedding, LSTM, PReLU, Dense
from sklearn.metrics import classification_report
from keras.optimizers import SGD
import numpy as np
from data.news_classification import create_vocabulary, read_news_corpus
from text_classification import data2Vecter, batch, corpus_filter

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries_dir", type=str, default="/tmp/rnntext",
                        help="Path to save summary logs for TensorBoard.")
    parser.add_argument("--epoches", type=int, default=20, help="epoches")
    parser.add_argument("--num_classes", type=int, default=18, help="the nums of classify")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate for opt")
    parser.add_argument("--num_sampled", type=int, default=5, help="samples")
    parser.add_argument("--batch_size", type=int, default=8, help="each batch contains samples")
    parser.add_argument("--decay_steps", type=int, default=1000, help="each steps decay the lr")
    parser.add_argument("--decay_rate", type=float, default=0.9, help="the decay rate for lr")
    parser.add_argument("--sequence_length", type=int, default=30, help="sequence length")
    parser.add_argument("--vocab_size", type=int, default=150346, help="the num of vocabs")
    parser.add_argument("--embed_size", type=int, default=200, help="embedding size")
    parser.add_argument("--is_training", type=bool, default=True, help='training or not')
    parser.add_argument("--keep_prob", type=float, default=0.7, help='keep prob')
    parser.add_argument("--clip_gradients", type=float, default=5.0, help='clip gradients')
    parser.add_argument("--filter_sizes", type=list, default=[2, 3, 4], help='filter size')
    parser.add_argument("--num_filters", type=int, default=128, help='num filters')

    return parser.parse_known_args()


FLAGS, unparsed = args()


class Attention(Layer):

    def __init__(self, attention_size, **kwargs):
        self.attention_size = attention_size
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        print("input shape: ", input_shape)

        # W: (EMBED_SIZE, ATTENTION_SIZE)
        # b: (ATTENTION_SIZE, 1)
        # u: (ATTENTION_SIZE, 1)
        self.W = self.add_weight(name="W_{:s}".format(self.name),
                                 shape=(input_shape[-1], self.attention_size),
                                 initializer="glorot_normal",
                                 trainable=True)
        self.b = self.add_weight(name="b_{:s}".format(self.name),
                                 shape=(input_shape[1], 1),
                                 initializer="zeros",
                                 trainable=True)
        self.u = self.add_weight(name="u_{:s}".format(self.name),
                                 shape=(self.attention_size, 1),
                                 initializer="glorot_normal",
                                 trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x, mask=None, **kwargs):
        # input: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        # input: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        # et: (BATCH_SIZE, MAX_TIMESTEPS, ATTENTION_SIZE)
        et = K.tanh(K.dot(x, self.W) + self.b)
        # at: (BATCH_SIZE, MAX_TIMESTEPS)
        at = K.softmax(K.squeeze(K.dot(et, self.u), axis=-1))
        if mask is not None:
            at *= K.cast(mask, K.floatx())
        # ot: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        atx = K.expand_dims(at, axis=-1)
        ot = atx * x
        # output: (BATCH_SIZE, EMBED_SIZE)
        output = K.sum(ot, axis=1)
        return output

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


inputs = Input(shape=(FLAGS.sequence_length,), dtype='float64')
# 第1层：Embedding层，one_hot转化成256维词向量
x = Embedding(input_dim=FLAGS.vocab_size, output_dim=256, mask_zero=True, name='embed')(inputs)
# 第2层：LSTM，64个特征
h1 = LSTM(64, return_sequences=True)(x)
# h2 = LSTM(64)(h1)
# model.add(PReLU())
attention = Attention(attention_size=110)(h1)
# 第3层：7个类别，全连接
outputs = Dense(FLAGS.num_classes, activation='softmax')(attention)
model = Model(inputs, outputs)

sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

if __name__ == '__main__':
    # -----------------  准备数据  ------------------
    vocab2int, int2vocab, label2int, int2label = create_vocabulary()
    train_corpus, train_labels, test_corpus, test_labels, dev_corpus, dev_labels = read_news_corpus()

    labels = ['discovery', 'story', 'essay']
    train_corpus, train_labels, test_corpus, test_labels, dev_corpus, dev_labels = corpus_filter(train_corpus,
                                                                                                 train_labels,
                                                                                                 test_corpus,
                                                                                                 test_labels,
                                                                                                 dev_corpus,
                                                                                                 dev_labels,
                                                                                                 labels=labels)

    train_X, train_y = data2Vecter(train_corpus, train_labels, vocab2int, label2int,
                                   sequence_length=FLAGS.sequence_length, sparse=False)
    test_X, test_y = data2Vecter(test_corpus, test_labels, vocab2int, label2int,
                                 sequence_length=FLAGS.sequence_length,
                                 sparse=False)
    dev_X, dev_y = data2Vecter(dev_corpus, dev_labels, vocab2int, label2int, sequence_length=FLAGS.sequence_length,
                               sparse=False)

    train_X, train_y = np.array(train_X), np.array(train_y)
    test_X, test_ = np.array(test_X), np.array(test_y)

    # -----------------  准备数据  ------------------
    model.fit(train_X, train_y, batch_size=64, epochs=20, validation_split=0.2, verbose=1)
