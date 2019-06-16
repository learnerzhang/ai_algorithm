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
from sklearn.model_selection import GridSearchCV
from data.news_classification import read_news_corpus
from text_classification import corpus_filter

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

train_corpus, train_labels, test_corpus, test_labels, dev_corpus, dev_labels = read_news_corpus()

labels = ['discovery', 'story', 'essay']
train_corpus, train_labels, test_corpus, test_labels, dev_corpus, dev_labels = corpus_filter(train_corpus, train_labels,
                                                                                             test_corpus, test_labels,
                                                                                             dev_corpus, dev_labels,
                                                                                             labels=labels)

X_train = [" ".join(dat) for dat in train_corpus]
X_test = [" ".join(dat) for dat in test_corpus]
X_dev = [" ".join(dat) for dat in dev_corpus]

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

# 调用 GridSearchCV，将 SVC(), tuned_parameters, cv=5, 还有 scoring 传递进去，
clf = GridSearchCV(svm.SVC(), param_grid={'kernel': ['linear'], 'C': [1]}, cv=5, scoring='precision_macro')
# 用训练集训练这个学习器 clf
clf.fit(X_train, y_train)

print("Best parameters set found on development set:")
print()

# 再调用 clf.best_params_ 就能直接得到最好的参数搭配结果
print(clf.best_params_)

print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']

# 看一下具体的参数间不同数值的组合后得到的分数是多少
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full train set.")
print("The scores are computed on the full train set.", clf.score(X_train, y_train))
print()
y_train_true, y_train_pred = y_train, clf.predict(X_train)
print(classification_report(y_train_true, y_train_pred))
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.", clf.score(X_dev, y_dev))
print()
y_dev_true, y_dev_pred = y_dev, clf.predict(X_dev)
print(classification_report(y_dev_true, y_dev_pred))

print()
print("The scores are computed on the full evaluation set.", clf.score(X_test, y_test))
print()
y_true, y_pred = y_test, clf.predict(X_test)
# 打印在测试集上的预测结果与真实值的分数
print(classification_report(y_true, y_pred))
