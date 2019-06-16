# -*- coding: utf-8 -*-
import argparse

import tensorflow as tf
import numpy as np

from data.news_classification import read_news_corpus, create_vocabulary
from text_classification import data2Vecter, batch, corpus_filter


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries_dir", type=str, default="/tmp/fasttext",
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

    return parser.parse_known_args()


FLAGS, unparsed = args()


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


class fastText(object):
    def __init__(self):
        """init all hyperparameter here"""
        # set hyper parameter
        self.label_size = FLAGS.num_classes
        self.batch_size = FLAGS.batch_size
        self.num_sampled = FLAGS.num_sampled
        self.sentence_length = FLAGS.sequence_length
        self.vocab_size = FLAGS.vocab_size
        self.embed_size = FLAGS.embed_size
        self.is_training = FLAGS.is_training
        self.learning_rate = FLAGS.lr

        # add placeholder (X,label)
        self.sentence = tf.placeholder(tf.int32, [None, self.sentence_length], name="sentence")  # X
        self.labels = tf.placeholder(tf.int32, [None], name="Labels")  # y

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = FLAGS.decay_steps, FLAGS.decay_rate

        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.instantiate_weights()
        self.logits = self.inference()  # [None, self.label_size]
        if not FLAGS.is_training:
            return
        self.loss_val = self.loss()
        self.train_op = self.train()
        self.predictions = tf.argmax(self.logits, axis=1, name="predictions")  # shape:[None,]

        # tf.argmax(self.logits, 1)-->[batch_size]
        correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")  # shape=()
        tf.summary.scalar("accuracy", self.accuracy)

    def instantiate_weights(self):
        """define all weights here"""
        # embedding matrix
        with tf.name_scope("embeddings"):
            self.Embedding = tf.get_variable("Embedding", [self.vocab_size, self.embed_size])
            variable_summaries(self.Embedding)

        with tf.name_scope("weights"):
            self.W = tf.get_variable("W", [self.embed_size, self.label_size])
            variable_summaries(self.W)
        with tf.name_scope("biases"):
            self.b = tf.get_variable("b", [self.label_size])
            variable_summaries(self.b)

    def inference(self):
        """main computation graph here: 1.embedding-->2.average-->3.linear classifier"""
        # 1.get emebedding of words in the sentence
        # [None,self.sentence_len,self.embed_size]
        sentence_embeddings = tf.nn.embedding_lookup(self.Embedding, self.sentence)

        # 2.average vectors, to get representation of the sentence
        # [None,self.embed_size]
        self.sentence_embeddings = tf.reduce_mean(sentence_embeddings, axis=1)

        # 3.linear classifier layer
        logits = tf.matmul(self.sentence_embeddings, self.W) + self.b
        # [None, self.label_size]==tf.matmul([None,self.embed_size],[self.embed_size,self.label_size])
        tf.summary.histogram("logists", logits)
        return logits

    def loss(self, l2_lambda=0.01):  # 0.0001-->0.001
        """calculate loss using (NCE)cross entropy here"""
        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        if self.is_training:  # training
            labels = tf.reshape(self.labels, [-1])  # [batch_size,1]------>[batch_size,]
            labels = tf.expand_dims(labels, 1)  # [batch_size,]----->[batch_size,1]
            with tf.name_scope("loss"):
                loss = tf.reduce_mean(
                    # The forward activations of the input network.
                    # inputs: A `Tensor` of shape `[batch_size, dim]`.
                    tf.nn.nce_loss(weights=tf.transpose(self.W), biases=self.b, labels=labels,
                                   inputs=self.sentence_embeddings, num_sampled=self.num_sampled,
                                   num_classes=self.label_size, partition_strategy="div"))
                tf.summary.scalar("loss", loss)
        else:  # eval/inference
            # logits = tf.matmul(self.sentence_embeddings, tf.transpose(self.W)) #matmul([None,self.embed_size])--->
            # logits = tf.nn.bias_add(logits, self.b)
            # [batch_size]---->[batch_size,label_size]
            labels_one_hot = tf.one_hot(self.labels, self.label_size)
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_one_hot, logits=self.logits)
            print("loss0:", loss)  # shape=(?, 1999)
            loss = tf.reduce_sum(loss, axis=1)
            print("loss1:", loss)  # shape=(?,)
        return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                   self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,
                                                   learning_rate=learning_rate, optimizer="Adam")
        return train_op


if __name__ == '__main__':
    fastText = fastText()
    print("FastText", fastText)

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
        # summary
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')
        sess.run(tf.global_variables_initializer())

        step_count = 0
        for epoch in range(FLAGS.epoches):
            for i, (b_x, b_y) in enumerate(batch(train_X, train_y, batch_size=32)):
                step_count += 1
                # [None, self.sequence_length]
                summary, loss, acc, predict, _ = sess.run(
                    [merged, fastText.loss_val, fastText.accuracy, fastText.predictions, fastText.train_op],
                    feed_dict={fastText.sentence: b_x, fastText.labels: b_y})
                train_writer.add_summary(summary, step_count)  # 训练数据集产生的

                if i % 20 == 0:
                    print("<>Train epoch: {} > step:{} | loss:{} | acc: {} ".format(epoch + 1, step_count, loss, acc))

                    test_summary, test_loss, test_acc = sess.run([merged, fastText.loss_val, fastText.accuracy],
                                                                 feed_dict={fastText.sentence: test_X, fastText.labels: test_y})
                    test_writer.add_summary(test_summary, step_count)  # 训练数据集产生的
                    print("<>Test epoch: {} > step:{} | loss:{} | acc: {} ".format(epoch + 1, step_count, test_loss, test_acc))

            dev_loss, dev_acc = sess.run([fastText.loss_val, fastText.accuracy],
                                         feed_dict={fastText.sentence: dev_X, fastText.labels: dev_y})
            print("<>DEV epoch: {} | loss:{} | acc: {} ".format(epoch + 1, dev_loss, dev_acc))
