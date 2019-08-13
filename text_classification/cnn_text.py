#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-06-05 17:55
# @Author  : zhangzhen
# @Site    : 
# @File    : cnn_text.py
# @Software: PyCharm
import logging
import argparse
import tensorflow as tf
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
    parser.add_argument("--summaries_dir", type=str, default="/tmp/fasttext_logs",
                        help="Path to save summary logs for TensorBoard.")
    parser.add_argument("--epoches", type=int, default=12, help="epoches")
    parser.add_argument("--num_classes", type=int, default=18, help="the nums of classify")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate for opt")
    parser.add_argument("--num_sampled", type=int, default=5, help="samples")
    parser.add_argument("--batch_size", type=int, default=8, help="each batch contains samples")
    parser.add_argument("--decay_steps", type=int, default=1000, help="each steps decay the lr")
    parser.add_argument("--decay_rate", type=float, default=0.9, help="the decay rate for lr")
    parser.add_argument("--sequence_length", type=int, default=30, help="sequence length")
    parser.add_argument("--vocab_size", type=int, default=150346, help="the num of vocabs")
    parser.add_argument("--embed_size", type=int, default=100, help="embedding size")
    parser.add_argument("--is_training", type=bool, default=True, help='training or not')
    parser.add_argument("--keep_prob", type=float, default=0.9, help='keep prob')
    parser.add_argument("--clip_gradients", type=float, default=5.0, help='clip gradients')
    parser.add_argument("--filter_sizes", type=list, default=[2, 3, 4], help='filter size')
    parser.add_argument("--num_filters", type=int, default=128, help='num filters')

    return parser.parse_known_args()


FLAGS, unparsed = args()


class TextCNN(object):

    def __init__(self, initializer=tf.random_normal_initializer(stddev=0.1)):
        # set hyperparamter
        self.num_classes = FLAGS.num_classes
        self.batch_size = FLAGS.batch_size
        self.sequence_length = FLAGS.sequence_length
        self.vocab_size = FLAGS.vocab_size
        self.embed_size = FLAGS.embed_size
        self.is_training = FLAGS.is_training
        self.learning_rate = tf.Variable(FLAGS.lr, trainable=False, name="learning_rate")  # ADD learning_rate
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * FLAGS.decay_rate)
        self.filter_sizes = FLAGS.filter_sizes  # it is a list of int. e.g. [3,4,5]
        self.num_filters = FLAGS.num_filters
        self.initializer = initializer
        self.num_filters_total = self.num_filters * len(FLAGS.filter_sizes)  # how many filters totally.
        self.clip_gradients = FLAGS.clip_gradients

        # add placeholder (X,label)
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")  # X
        self.input_y = tf.placeholder(tf.int32, [None], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.iter = tf.placeholder(tf.int32)  # training iteration
        self.tst = tf.placeholder(tf.bool)

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.b1 = tf.Variable(tf.ones([self.num_filters]) / 10)
        self.b2 = tf.Variable(tf.ones([self.num_filters]) / 10)
        self.decay_steps, self.decay_rate = FLAGS.decay_steps, FLAGS.decay_rate

        self.instantiate_weights()
        self.logits = self.inference()  # [None, self.label_size]. main computation graph is here.
        self.possibility = tf.nn.sigmoid(self.logits)
        if not FLAGS.is_training:
            return
        self.loss_val = self.loss()
        self.train_op = self.train()

        # shape:[None,]
        self.predictions = tf.argmax(self.logits, axis=1, name="predictions")
        # tf.argmax(self.logits, 1)-->[batch_size]
        correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.input_y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")  # shape=()
        tf.summary.scalar("accuracy", self.accuracy)

    def instantiate_weights(self):
        """define all weights here"""
        with tf.name_scope("embedding"):  # embedding matrix
            # [vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],
                                             initializer=self.initializer)
            variable_summaries(self.Embedding)

        with tf.name_scope("weights"):
            # [embed_size,label_size]
            self.W_projection = tf.get_variable("W_projection", shape=[self.num_filters_total, self.num_classes],
                                                initializer=self.initializer)
            variable_summaries(self.W_projection)
        with tf.name_scope("biases"):
            # [label_size] #ADD 2017.06.09
            self.b_projection = tf.get_variable("b_projection", shape=[self.num_classes])
            variable_summaries(self.b_projection)

    def inference(self):

        # [None,sentence_length,embed_size]
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding, self.input_x)

        # [None,sentence_length,embed_size,1). expand dimension so meet input requirement of 2d-conv
        self.sentence_embeddings_expanded = tf.expand_dims(self.embedded_words, -1)

        logging.debug("embedded_words %s" % self.embedded_words)
        logging.debug("sentence_embeddings_expanded %s" % self.sentence_embeddings_expanded)

        # conv2d
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("convolution-pooling-%s" % filter_size):
                filter = tf.get_variable("filter-%s" % filter_size, [filter_size, self.embed_size, 1, self.num_filters],
                                         initializer=self.initializer)
                conv = tf.nn.conv2d(self.sentence_embeddings_expanded, filter, strides=[1, 1, 1, 1], padding="VALID",
                                    name="conv")
                conv, self.update_ema = self.batchnorm(conv, self.tst, self.iter, self.b1)  # TODO remove it temp

                b = tf.get_variable("b-%s" % filter_size, [self.num_filters])  # ADD 2017-06-09
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                pooled = tf.nn.max_pool(h, ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding='VALID', name="pool")
                logging.debug('pooled: %s' % pooled)
                pooled_outputs.append(pooled)

        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.num_filters_total])
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, keep_prob=self.dropout_keep_prob)  # [None,num_filters_total]
        self.h_drop = tf.layers.dense(self.h_drop, self.num_filters_total, activation=tf.nn.tanh, use_bias=True)

        with tf.name_scope("output"):
            # shape:[None, self.num_classes]==tf.matmul([None,self.embed_size],[self.embed_size,self.num_classes])
            logits = tf.matmul(self.h_drop, self.W_projection) + self.b_projection
            tf.summary.histogram("logists", logits)
        return logits

    def batchnorm(self, Ylogits, is_test, iteration, offset,
                  convolutional=False):  # check:https://github.com/martin-gorner/tensorflow-mnist-tutorial/blob/master/mnist_4.1_batchnorm_five_layers_relu.py#L89
        """
        batch normalization: keep moving average of mean and variance. use it as value for BN when training. when prediction, use value from that batch.
        :param Ylogits:
        :param is_test:
        :param iteration:
        :param offset:
        :param convolutional:
        :return:
        """
        # adding the iteration prevents from averaging across non-existing iterations
        exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration)
        bnepsilon = 1e-5
        if convolutional:
            mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
        else:
            mean, variance = tf.nn.moments(Ylogits, [0])
        update_moving_averages = exp_moving_avg.apply([mean, variance])
        m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
        v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
        Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
        return Ybn, update_moving_averages

    def loss(self, l2_lambda=0.0001):  # 0.001
        with tf.name_scope("loss"):
            print("Label: ", self.input_y, "Logits: ", self.logits)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            loss = tf.reduce_mean(losses)
            l2_losses = tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_losses
        return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                   self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,
                                                   learning_rate=learning_rate, optimizer="Adam",
                                                   clip_gradients=self.clip_gradients)
        return train_op


if __name__ == '__main__':
    textCNN = TextCNN()

    vocab2int, int2vocab, label2int, int2label = create_vocabulary()
    train_corpus, train_labels, test_corpus, test_labels, dev_corpus, dev_labels = read_news_corpus()
    labels = ['discovery', 'story', 'essay']

    train_corpus, train_labels, test_corpus, test_labels, dev_corpus, dev_labels = corpus_filter(train_corpus,
                                                                                                 train_labels,
                                                                                                 test_corpus,
                                                                                                 test_labels,
                                                                                                 dev_corpus, dev_labels,
                                                                                                 labels=labels)

    train_X, train_y = data2Vecter(train_corpus, train_labels, vocab2int, label2int,
                                   sequence_length=FLAGS.sequence_length)
    test_X, test_y = data2Vecter(test_corpus, test_labels, vocab2int, label2int, sequence_length=FLAGS.sequence_length)
    dev_X, dev_y = data2Vecter(dev_corpus, dev_labels, vocab2int, label2int, sequence_length=FLAGS.sequence_length)

    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('/tmp/textcnn/train', sess.graph)
        test_writer = tf.summary.FileWriter('/tmp/textcnn/test')
        sess.run(tf.global_variables_initializer())
        step_count = 0
        best = 0
        for epoch in range(FLAGS.epoches):
            for i, (b_x, b_y) in enumerate(batch(train_X, train_y, batch_size=32, seed=1111)):
                step_count += 1
                # b_x： (batch_size, sequence_length)
                # b_y： (batch_size, )
                # logging.info("({}, {}), {}".format(len(b_x), len(b_x[0]), len(b_y)))
                summary, loss, possibility, W_projection_value, acc, _ = sess.run(
                    [merged, textCNN.loss_val, textCNN.possibility, textCNN.W_projection, textCNN.accuracy,
                     textCNN.train_op],
                    feed_dict={textCNN.input_x: b_x,
                               textCNN.input_y: b_y,
                               textCNN.dropout_keep_prob: FLAGS.keep_prob,
                               textCNN.tst: False})
                train_writer.add_summary(summary, step_count)
                # logging.info('%sth iter %s> loss: %s' % (i, 10 * '-', loss))

                if i % 20 == 0:
                    logging.info("<>Train epoch: {} > step:{} | loss:{} | acc: {} ".format(epoch + 1, step_count, loss, acc))

                    test_summary, test_loss, test_acc = sess.run([merged, textCNN.loss_val, textCNN.accuracy],
                                                                 feed_dict={textCNN.input_x: test_X,
                                                                            textCNN.input_y: test_y,
                                                                            textCNN.dropout_keep_prob: 1.0,
                                                                            textCNN.tst: True})
                    test_writer.add_summary(test_summary, step_count)  # 训练数据集产生的
                    logging.info("<>Test epoch: {} > step:{} | loss:{} | acc: {} ".format(epoch + 1, step_count, test_loss, test_acc))

            dev_loss, dev_acc = sess.run([textCNN.loss_val, textCNN.accuracy],
                                         feed_dict={textCNN.input_x: dev_X, textCNN.input_y: dev_y,
                                                    textCNN.dropout_keep_prob: 1.0,
                                                    textCNN.tst: True})
            print("<>DEV epoch: {} | loss:{} | acc: {} ".format(epoch + 1, dev_loss, dev_acc))
