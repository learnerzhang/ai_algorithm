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


def extract_argmax_and_embed(embedding, output_projection=None):
    def loop_function(prev, _):
        if output_projection is not None:
            prev = tf.matmul(prev, output_projection[0]) + output_projection[1]
        prev_symbol = tf.argmax(prev, 1)
        emb_prev = tf.gather(embedding, prev_symbol)
        return emb_prev

    return loop_function


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
    parser.add_argument("--decoder_sent_length", type=int, default=6, help='decoder sent length')
    parser.add_argument("--hidden_size", type=int, default=6, help='hidden size')
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
        self.hidden_size = FLAGS.embed_size
        self.is_training = FLAGS.is_training
        self.learning_rate = tf.Variable(FLAGS.lr, trainable=False, name='learning_rate')

        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * 0.5)
        self.initializer = initializer

        self.decoder_sent_length = FLAGS.decoder_sent_length
        self.hidden_size = FLAGS.hidden_size
        self.clip_gradients = FLAGS.clip_gradients
        self.l2_lambda = FLAGS.l2_lambda

        self.input_x = tf.placeholder(tf.int32, [None, FLAGS.sequence_length], name="input_x")  # x
        self.decoder_input = tf.placeholder(tf.int32, [None, FLAGS.decoder_sent_length],
                                            name="decoder_input")  # y, but shift
        self.input_y_label = tf.placeholder(tf.int32, [None, FLAGS.decoder_sent_length],
                                            name="input_y_label")  # y, but shift
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = FLAGS.decay_steps, FLAGS.decay_rate

        self.instantiate_weights()
        self.logits = self.inference()

        self.predictions = tf.argmax(self.logits, axis=2, name="predictions")
        self.loss_val = self.loss_seq2seq()
        self.train_op = self.train()

    def gru_cell(self, Xt, h_t_minus_1):
        """
        RNN 单元结构
        :param Xt:
        :param h_t_minus_1:
        :return:
        """
        z_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_z) + tf.matmul(h_t_minus_1, self.U_z) + self.b_z)
        r_t = tf.nn.sigmoid(tf.matmul(Xt, self.W_r) + tf.matmul(h_t_minus_1, self.U_r) + self.b_r)

        h_t_candiate = tf.nn.tanh(tf.matmul(Xt, self.W_h) + r_t * (tf.matmul(h_t_minus_1, self.U_h)) + self.b_h)
        h_t = (1 - z_t) * h_t_minus_1 + z_t * h_t_candiate
        return h_t, h_t

    def gru_forward(self, embedded_words, gru_cell, reverse=False):
        embedded_words_splitted = tf.split(embedded_words, self.sequence_length, axis=1)
        embedded_words_squeeze = [tf.squeeze(x, axis=1) for x in embedded_words_splitted]

        h_t = tf.ones((self.batch_size, self.hidden_size))
        h_t_list = []
        if reverse:
            embedded_words_squeeze.reverse()

        for time_step, Xt in enumerate(embedded_words_squeeze):
            h_t = gru_cell(Xt, h_t)
            h_t_list.append(h_t)

        if reverse:
            h_t_list.reverse()
        return h_t_list

    def inference(self):
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding, self.input_x)
        # 正向
        hidden_state_forward_list = self.gru_forward(self.embedded_words, self.gru_cell)
        # 反向
        hidden_state_backward_list = self.gru_forward(self.embedded_words, self.gru_cell, reverse=True)

        thought_vector_list = [tf.concat([h_forward, h_backward], axis=1) for h_forward, h_backward in
                               zip(hidden_state_forward_list, hidden_state_backward_list)]

        thought_vector = tf.stack(thought_vector_list, axis=1)
        initial_state = tf.nn.tanh(
            tf.matmul(hidden_state_backward_list[0], self.W_initial_state) + self.b_initial_state)

        cell = self.gru_cell_decoder
        output_projection = (self.W_projection, self.b_projection)

        loop_function = extract_argmax_and_embed(self.Embedding_label,
                                                 output_projection) if not self.is_training else None

    def instantiate_weights(self):

        with tf.name_scope("decoder_init_state"):
            self.W_initial_state = tf.get_variable("W_initial_state", shape=[self.hidden_size, self.hidden_size * 2], initializer=self.initializer)
            self.b_initial_state = tf.get_variable("b_initial_state", shape=[self.hidden_size * 2])

        with tf.name_scope("embedding_projection"):
            # [vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size], initializer=self.initializer)
            # ,initializer=self.initializer
            self.Embedding_label = tf.get_variable("Embedding_label", shape=[self.num_classes, self.embed_size * 2], dtype=tf.float32)
            # [embed_size,label_size]
            self.W_projection = tf.get_variable("W_projection", shape=[self.hidden_size * 2, self.num_classes], initializer=self.initializer)
            self.b_projection = tf.get_variable("b_projection", shape=[self.num_classes])

        with tf.name_scope("gru_weights_encoder"):
            self.W_z = tf.get_variable("W_z", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.U_z = tf.get_variable("U_z", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.b_z = tf.get_variable("b_z", shape=[self.hidden_size])
            # GRU parameters:reset gate related
            self.W_r = tf.get_variable("W_r", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.U_r = tf.get_variable("U_r", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.b_r = tf.get_variable("b_r", shape=[self.hidden_size])

            self.W_h = tf.get_variable("W_h", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.U_h = tf.get_variable("U_h", shape=[self.embed_size, self.hidden_size], initializer=self.initializer)
            self.b_h = tf.get_variable("b_h", shape=[self.hidden_size])


if __name__ == '__main__':
    textRNN = TextAttRNN()
    # vocab2int, int2vocab, label2int, int2label = create_vocabulary()
    # train_corpus, train_labels, test_corpus, test_labels, dev_corpus, dev_labels = read_news_corpus()
    #
    # labels = ['discovery', 'story', 'essay']
    # train_corpus, train_labels, test_corpus, test_labels, dev_corpus, dev_labels = corpus_filter(train_corpus,
    #                                                                                              train_labels,
    #                                                                                              test_corpus,
    #                                                                                              test_labels,
    #                                                                                              dev_corpus, dev_labels,
    #                                                                                              labels=labels)
    #
    # train_X, train_y = data2Vecter(train_corpus, train_labels, vocab2int, label2int,
    #                                sequence_length=FLAGS.sequence_length)
    # test_X, test_y = data2Vecter(test_corpus, test_labels, vocab2int, label2int, sequence_length=FLAGS.sequence_length)
    # dev_X, dev_y = data2Vecter(dev_corpus, dev_labels, vocab2int, label2int, sequence_length=FLAGS.sequence_length)
    #
    # with tf.Session() as sess:
    #     merged = tf.summary.merge_all()
    #     train_writer = tf.summary.FileWriter('/tmp/rnn_att_text/train', sess.graph)
    #     test_writer = tf.summary.FileWriter('/tmp/rnn_att_text/test')
    #     sess.run(tf.global_variables_initializer())
    #
    #     step_count = 0
    #     for epoch in range(FLAGS.epoches):
    #         for i, (b_x, b_y) in enumerate(batch(train_X, train_y, batch_size=32)):
    #             step_count += 1
    #             # b_x： (batch_size, sequence_length)
    #             # b_y： (batch_size, )
    #             summary, loss, W_projection_value, acc, _ = sess.run(
    #                 [merged, textRNN.loss_val, textRNN.W_projection, textRNN.accuracy,
    #                  textRNN.train_op],
    #                 feed_dict={textRNN.input_x: b_x,
    #                            textRNN.input_y: b_y,
    #                            textRNN.dropout_keep_prob: FLAGS.keep_prob})
    #             train_writer.add_summary(summary, step_count)
    #             # logging.info('%sth iter %s> loss: %s' % (i, 10 * '-', loss))
    #
    #             if i % 10 == 0:
    #                 logging.info(
    #                     "<>Train epoch: {} > step:{} | loss:{} | acc: {} ".format(epoch + 1, step_count, loss, acc))
    #                 test_summary, test_loss, test_acc = sess.run([merged, textRNN.loss_val, textRNN.accuracy],
    #                                                              feed_dict={textRNN.input_x: test_X,
    #                                                                         textRNN.input_y: test_y,
    #                                                                         textRNN.dropout_keep_prob: 1.0, })
    #                 test_writer.add_summary(test_summary, step_count)  # 训练数据集产生的
    #                 logging.info(
    #                     "<>Test epoch: {} > step:{} | loss:{} | acc: {} ".format(epoch + 1, step_count, test_loss,
    #                                                                              test_acc))
    #
    #         dev_loss, dev_acc = sess.run([textRNN.loss_val, textRNN.accuracy],
    #                                      feed_dict={textRNN.input_x: dev_X, textRNN.input_y: dev_y,
    #                                                 textRNN.dropout_keep_prob: 1.0, })
    #         print("<>DEV epoch: {} | loss:{} | acc: {} ".format(epoch + 1, dev_loss, dev_acc))
