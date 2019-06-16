#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-06-04 15:00
# @Author  : zhangzhen
# @Site    : 
# @File    : multinomial_nb_text.py
# @Software: PyCharm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB

from data.news_classification import read_news_corpus
from text_classification import corpus_filter

labels = ['discovery', 'story', 'essay']

train_corpus, train_labels, test_corpus, test_labels, dev_corpus, dev_labels = read_news_corpus()

train_corpus, train_labels, test_corpus, test_labels, dev_corpus, dev_labels = corpus_filter(train_corpus, train_labels,
                                                                                             test_corpus, test_labels,
                                                                                             dev_corpus, dev_labels,
                                                                                             labels=labels)

X_train = ["".join(dat) for dat in train_corpus]
X_test = ["".join(dat) for dat in test_corpus]
X_dev = ["".join(dat) for dat in dev_corpus]

y_train = train_labels
y_test = test_labels
y_dev = dev_labels

# 文本特征向量化
vec = CountVectorizer()
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)
X_dev = vec.transform(X_dev)

print("d:", X_train[0])

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
