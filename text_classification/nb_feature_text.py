#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-06-04 15:00
# @Author  : zhangzhen
# @Site    : 
# @File    : multinomial_nb_text.py
# @Software: PyCharm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from matplotlib import pyplot as plt
import numpy as np
from data.news_classification import read_news_corpus
from text_classification import corpus_filter
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

labels = ['discovery', 'story', 'essay']
train_corpus, train_labels, test_corpus, test_labels, dev_corpus, dev_labels = read_news_corpus()
train_corpus, train_labels, test_corpus, test_labels, dev_corpus, dev_labels = corpus_filter(train_corpus, train_labels,
                                                                                             test_corpus, test_labels,
                                                                                             dev_corpus, dev_labels,
                                                                                             labels=labels)

X_train = [" ".join(dat) for dat in train_corpus]
X_test = [" ".join(dat) for dat in test_corpus]
X_dev = [" ".join(dat) for dat in dev_corpus]

# tfidf_model = TfidfVectorizer().fit(X_train)
tfidf_model = TfidfVectorizer(min_df=0.002, max_df=0.5, token_pattern=r"(?u)\b\w\w+\b").fit(X_train)

# 得到tf-idf矩阵，稀疏矩阵表示法
X_train = tfidf_model.transform(X_train).todense()
X_test = tfidf_model.transform(X_test).todense()
X_dev = tfidf_model.transform(X_dev).todense()

y_train = train_labels
y_test = test_labels
y_dev = dev_labels

print("Train shape:", X_train.shape, "\nTest shape:", X_test.shape, "\nDev shape:", X_dev.shape)
# 词语与列的对应关系
print(tfidf_model.vocabulary_)


def train():
    # 模型
    nb = MultinomialNB()  # 使用默认配置初始化朴素贝叶斯
    nb.fit(X_train, y_train)  # 利用训练数据对模型参数进行估计
    y_predict = nb.predict(X_test)  # 对参数进行预测
    y_dev_predict = nb.predict(X_dev)  # 对参数进行预测

    # 获取结果报告
    print()
    print('The Test Accuracy of Naive Bayes Classifier is:', nb.score(X_test, y_test))
    print(classification_report(y_test, y_predict, target_names=labels))
    print()
    print('The Dev Accuracy of Naive Bayes Classifier is:', nb.score(X_dev, y_dev))
    print(classification_report(y_dev, y_dev_predict, target_names=labels))


def search_best_alpha():
    alphas = np.linspace(0, 1, num=1000)
    # alphas = np.logspace(-5, 2, num=100)
    training_scores = []
    testing_scores = []
    for alpha in alphas:
        clf = MultinomialNB(alpha=alpha)
        clf.fit(X_train, y_train)
        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)
        logging.info("alpha: {}, train score: {}, test score: {}".format(alpha, train_score, test_score))

        training_scores.append(train_score)
        testing_scores.append(test_score)

    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(alphas, training_scores, label="training score", marker='o')
    ax.plot(alphas, testing_scores, label="testing score", marker='*')
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("score")
    ax.set_title("MultinomialNB Classification")
    ax.legend(framealpha=0.5, loc='best')
    plt.show()


if __name__ == '__main__':
    search_best_alpha()
