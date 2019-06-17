#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-06-06 15:00
# @Author  : zhangzhen
# @Site    : 
# @File    : rnntext.py
# @Software: PyCharm
import logging
import argparse
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
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
    parser.add_argument("--embed_size", type=int, default=300, help="embedding size")
    parser.add_argument("--is_training", type=bool, default=True, help='training or not')
    parser.add_argument("--keep_prob", type=float, default=0.7, help='keep prob')
    parser.add_argument("--clip_gradients", type=float, default=5.0, help='clip gradients')
    parser.add_argument("--filter_sizes", type=list, default=[2, 3, 4], help='filter size')
    parser.add_argument("--num_filters", type=int, default=128, help='num filters')
    parser.add_argument("--decoder_sent_length", type=int, default=6, help='decoder sent length')
    parser.add_argument("--hidden_size", type=int, default=100, help='hidden size')
    parser.add_argument("--l2_lambda", type=float, default=0.0001, help='hidden size')

    return parser.parse_known_args()


FLAGS, unparsed = args()


class TextAttRNN(object):

    def __init__(self, initializer=tf.random_normal_initializer(stddev=0.5)):
        """init all hyperparameter here"""
        # set hyperparamter
        self.num_classes = FLAGS.num_classes
        self.batch_size = FLAGS.batch_size
        self.sequence_length = FLAGS.sequence_length
        self.vocab_size = FLAGS.vocab_size
        self.embed_size = FLAGS.embed_size
        self.hidden_size = FLAGS.hidden_size
        self.is_training = FLAGS.is_training
        self.learning_rate = tf.Variable(FLAGS.lr, trainable=False, name='learning_rate')

        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * 0.5)
        self.initializer = initializer

        self.decoder_sent_length = FLAGS.decoder_sent_length
        self.hidden_size = FLAGS.hidden_size
        self.clip_gradients = FLAGS.clip_gradients
        self.l2_lambda = FLAGS.l2_lambda

        # add placeholder (X,label)
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")  # X
        # y: (None, )
        self.input_y = tf.placeholder(tf.int32, [None, ], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = FLAGS.decay_steps, FLAGS.decay_rate
        self.logits = self.inference()

        self.predictions = tf.argmax(self.logits, axis=1, name="predictions")  # shape:[None,]
        # tf.argmax(self.logits, 1)-->[batch_size]
        correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.input_y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")  # shape=()
        tf.summary.scalar("accuracy", self.accuracy)

        self.loss_val = self.loss()  # -->self.loss_nce()
        tf.summary.scalar("losses", self.loss_val)

        self.train_op = self.train()

    def inference(self):
        with tf.name_scope('embedding'):
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],
                                             initializer=self.initializer)
            self.embedded_inputs = tf.nn.embedding_lookup(self.Embedding, self.input_x)

        with tf.name_scope('biRNN'):
            # 正向
            cell_forward = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)
            cell_forward = tf.contrib.rnn.DropoutWrapper(cell_forward, self.dropout_keep_prob)
            # 反向
            cell_backward = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)
            cell_backward = tf.contrib.rnn.DropoutWrapper(cell_backward, self.dropout_keep_prob)

            output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_forward, cell_bw=cell_backward,
                                                        inputs=self.embedded_inputs,
                                                        # sequence_length=self.sequence_length,
                                                        dtype=tf.float32)

            logging.info("forward output: {}, backward output: {}".format(output[0].shape, output[1].shape))
            output = tf.concat(output, 2)
            logging.info("Bi-LSTM concat output: {}".format(output.shape))

        with tf.name_scope('attention'):
            r_list = []
            att_w = tf.Variable(tf.truncated_normal([self.hidden_size * 2, self.embed_size], stddev=0.1), name='att_w')
            att_u = tf.Variable(tf.truncated_normal([self.embed_size, 1], stddev=0.1), name='att_u')
            att_b = tf.Variable(tf.constant(0.1, shape=[self.embed_size]), name='att_b')
            logging.info("att_w: {}, att_u: {}, att_b: {}".format(att_w.shape, att_u.shape, att_b.shape))

            for t in range(self.sequence_length):
                # char -> (batch, embed_size)
                att_hidden = tf.tanh(tf.matmul(output[:, t, :], att_w) + tf.reshape(att_b, [1, -1]))
                att_out = tf.matmul(att_hidden, att_u)
                r_list.append(att_out)

            # total att out
            logit = tf.concat(r_list, axis=1)
            logging.info("sequences att out: {}".format(logit.shape))

            seq_weights = tf.nn.softmax(logit, name='att_softmax')
            logging.info("sequences att softmax out: {}".format(seq_weights.shape))

            # (batch_size, seq_length) -> (batch_size, seq_length, 1)
            seq_weights = tf.reshape(seq_weights, [-1, self.sequence_length, 1])

            # sum_0_29_{att_i * seq_dim_i} -> (batch_size, sum_dim)
            att_final_out = tf.reduce_sum(output * seq_weights, 1)
            logging.info("att shape: {}".format(seq_weights.shape))
            logging.info("final out shape: {}".format(att_final_out.shape))

        with tf.name_scope('dropout'):
            self.out_drop = tf.nn.dropout(att_final_out, keep_prob=self.dropout_keep_prob)

        with tf.name_scope('output'):
            w = tf.Variable(tf.truncated_normal([self.hidden_size * 2, self.num_classes], stddev=0.1), name='w')
            b = tf.Variable(tf.zeros([self.num_classes]), name='b')
            # (batch, hidden_size* 2) * (hidden_size, num_size)
            logits = tf.matmul(self.out_drop, w) + b

        return logits

    def loss(self, l2_lambda=0.001):
        with tf.name_scope("loss"):
            logging.info("input-y: {}, logist: {}".format(self.input_y, self.logits))
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            logging.info("losses: {}".format(losses))
            loss = tf.reduce_mean(losses)
            logging.info("loss: {}".format(loss))

            l2_losses = tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_losses
        return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                   self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,
                                                   learning_rate=learning_rate, optimizer="Adam")
        return train_op


if __name__ == '__main__':
    textRNN = TextAttRNN()
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
        train_writer = tf.summary.FileWriter('/tmp/rnn_att_text/train', sess.graph)
        test_writer = tf.summary.FileWriter('/tmp/rnn_att_text/test')
        sess.run(tf.global_variables_initializer())

        step_count = 0
        for epoch in range(FLAGS.epoches):
            for i, (b_x, b_y) in enumerate(batch(train_X, train_y, batch_size=32)):
                step_count += 1
                # b_x： (batch_size, sequence_length)
                # b_y： (batch_size, )
                summary, loss, acc, _ = sess.run([merged, textRNN.loss_val, textRNN.accuracy, textRNN.train_op],
                    feed_dict={textRNN.input_x: b_x,
                               textRNN.input_y: b_y,
                               textRNN.dropout_keep_prob: FLAGS.keep_prob})
                train_writer.add_summary(summary, step_count)
                # logging.info('%sth iter %s> loss: %s' % (i, 10 * '-', loss))

                if i % 10 == 0:
                    logging.info(
                        "<>Train epoch: {} > step:{} | loss:{} | acc: {} ".format(epoch + 1, step_count, loss, acc))
                    test_summary, test_loss, test_acc = sess.run([merged, textRNN.loss_val, textRNN.accuracy],
                                                                 feed_dict={textRNN.input_x: test_X,
                                                                            textRNN.input_y: test_y,
                                                                            textRNN.dropout_keep_prob: 1.0, })
                    test_writer.add_summary(test_summary, step_count)  # 训练数据集产生的
                    logging.info(
                        "<>Test epoch: {} > step:{} | loss:{} | acc: {} ".format(epoch + 1, step_count, test_loss,
                                                                                 test_acc))

            dev_loss, dev_acc = sess.run([textRNN.loss_val, textRNN.accuracy],
                                         feed_dict={textRNN.input_x: dev_X, textRNN.input_y: dev_y,
                                                    textRNN.dropout_keep_prob: 1.0, })
            print("<>DEV epoch: {} | loss:{} | acc: {} ".format(epoch + 1, dev_loss, dev_acc))
