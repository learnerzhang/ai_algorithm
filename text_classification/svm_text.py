#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-06-04 14:55
# @Author  : zhangzhen
# @Site    : 
# @File    : svm_text.py
# @Software: PyCharm
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn import svm
from data.news_classification import read_news_corpus
from text_classification import corpus_filter

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

tfidf_model = TfidfVectorizer().fit(X_train)

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

clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)  # 对参数进行预测
y_dev_predict = clf.predict(X_dev)  # 对参数进行预测

# 获取结果报告
print()
print('The Test Accuracy of Naive Bayes Classifier is:', clf.score(X_test, y_test))
print(classification_report(y_test, y_predict, target_names=labels))
print()
print('The Dev Accuracy of Naive Bayes Classifier is:', clf.score(X_dev, y_dev))
print(classification_report(y_dev, y_dev_predict, target_names=labels))
