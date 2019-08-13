#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-06-06 15:00
# @Author  : zhangzhen
# @Site    : 
# @File    : rnn_text.py
# @Software: PyCharm
import logging
import argparse
import tensorflow as tf
from keras import Sequential
from keras.layers import Embedding, LSTM, PReLU, Dense
from sklearn.metrics import classification_report
from keras.optimizers import SGD
import numpy as np
from data.news_classification import create_vocabulary, read_news_corpus
from text_classification import data2Vecter, batch, corpus_filter

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor"""
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.summary.scalar("mean", mean)

        with tf.name_scope("stddev"):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

        tf.summary.scalar("stddev", stddev)
        tf.summary.scalar("max", tf.reduce_mean(var))
        tf.summary.scalar("min", tf.reduce_min(var))
        tf.summary.histogram("histogram", var)


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


class KearasTextRNN(object):

    def __init__(self):
        """init all hyperparameter here"""
        # set hyperparamter
        # 建立LSTM模型
        model = Sequential()
        # 第1层：Embedding层，one_hot转化成256维词向量
        model.add(Embedding(input_dim=FLAGS.vocab_size, output_dim=256, mask_zero=True))

        # 第2层：LSTM，64个特征
        model.add(LSTM(64))
        model.add(PReLU())

        # 第3层：7个类别，全连接
        model.add(Dense(FLAGS.num_classes, activation='softmax'))
        sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        self.model = model

    def train(self, train_X, train_y, test_X, test_y):
        # 20%的数据作为验证集
        self.model.fit(train_X, train_y, batch_size=64, epochs=20, validation_split=0.2, verbose=1)

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_train_true, y_train_pred = train_y, self.model.predict(train_X, batch_size=10)
        print(classification_report(y_train_true, y_train_pred))

        print()
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = test_y, self.model.predict(test_X)
        # 打印在测试集上的预测结果与真实值的分数
        print(classification_report(y_true, y_pred))


if __name__ == '__main__':
    textRNN = KearasTextRNN()
    vocab2int, int2vocab, label2int, int2label = create_vocabulary()
    train_corpus, train_labels, test_corpus, test_labels, dev_corpus, dev_labels = read_news_corpus()

    labels = ['discovery', 'story', 'essay']
    train_corpus, train_labels, test_corpus, test_labels, dev_corpus, dev_labels = corpus_filter(train_corpus,
                                                                                                 train_labels,
                                                                                                 test_corpus,
                                                                                                 test_labels,
                                                                                                 dev_corpus, dev_labels,
                                                                                                 labels=labels)

    train_X, train_y = data2Vecter(train_corpus, train_labels, vocab2int, label2int, sequence_length=FLAGS.sequence_length, sparse=False)
    test_X, test_y = data2Vecter(test_corpus, test_labels, vocab2int, label2int, sequence_length=FLAGS.sequence_length, sparse=False)
    dev_X, dev_y = data2Vecter(dev_corpus, dev_labels, vocab2int, label2int, sequence_length=FLAGS.sequence_length,sparse=False)

    train_X, train_y = np.array(train_X), np.array(train_y)
    test_X, test_ = np.array(test_X), np.array(test_y)

    textRNN.train(train_X, train_y, test_X, test_y)
