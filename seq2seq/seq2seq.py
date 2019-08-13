#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-06-18 10:05
# @Author  : zhangzhen
# @Site    : 
# @File    : seq2seq.py
# @Software: PyCharm
import argparse
import logging
import tensorflow as tf

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
    parser.add_argument("--hidden_size", type=int, default=50, help="hidden size")
    parser.add_argument("--is_training", type=bool, default=True, help='training or not')
    parser.add_argument("--keep_prob", type=float, default=0.9, help='keep prob')
    parser.add_argument("--clip_gradients", type=float, default=5.0, help='clip gradients')
    parser.add_argument("--filter_sizes", type=list, default=[2, 3, 4], help='filter size')
    parser.add_argument("--num_filters", type=int, default=128, help='num filters')
    parser.add_argument("--rnn_layers", type=int, default=2, help='num layers for rnn')

    return parser.parse_known_args()


FLAGS, unparsed = args()


class Seq2Seq:

    def __init__(self, initializer=tf.random_normal_initializer(stddev=0.5)):
        self.num_classes = FLAGS.num_classes
        self.batch_size = FLAGS.batch_size
        self.sequence_length = FLAGS.sequence_length
        self.vocab_size = FLAGS.vocab_size
        self.embed_size = FLAGS.embed_size
        self.hidden_size = FLAGS.hidden_size
        self.rnn_layers = FLAGS.rnn_layers
        self.is_training = FLAGS.is_training
        self.learning_rate = FLAGS.lr
        self.initializer = initializer
        self.num_sampled = 10

        ## 定义输入
        self.inputs = tf.placeholder(tf.int32, shape=[None, self.sequence_length])

        self.logist = self.inference()

    def inference(self):
        def get_lstm_cell(num_units=128):
            """
            用于构建多层RNN神经网络
            :param num_units:
            :return:
            """
            return tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units)

        # Encoder
        with tf.name_scope('embedding'):
            encoder_embeddings = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embed_size], -1.0, 1.0, name='emcoder_embedding'))
            embed_inputs = tf.nn.embedding_lookup(encoder_embeddings, self.inputs)
            # 将原始输入转化成embedded输入，tf中在cpu进行

        with tf.name_scope('encoder'):
            # construct rnn_layers
            encoder_cells = tf.contrib.rnn.MultiRNNCell(
                [get_lstm_cell(num_units=self.hidden_size) for _ in range(self.rnn_layers)])
            logging.info("{},{}".format(encoder_cells.state_size, encoder_cells.output_size))
            logging.info("lstm unit :{}".format(encoder_cells))
            logging.info("lstm inputs :{}".format(embed_inputs))
            encoder_output, encoder_final_state = tf.nn.dynamic_rnn(encoder_cells, embed_inputs, dtype=tf.float32)
            logging.info("encoder output: {}".format(encoder_output))


if __name__ == '__main__':
    seq2seq = Seq2Seq()
