import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn import tree
from matplotlib import pyplot as plt
import numpy as np
from data.news_classification import read_news_corpus, create_vocabulary
from text_classification import corpus_filter

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

vocab2int, int2vocab, label2int, int2label = create_vocabulary()
train_corpus, train_labels, test_corpus, test_labels, dev_corpus, dev_labels = read_news_corpus()

labels = ['discovery', 'story', 'essay']

train_corpus, train_labels, test_corpus, test_labels, dev_corpus, dev_labels = corpus_filter(train_corpus, train_labels,
                                                                                             test_corpus, test_labels,
                                                                                             dev_corpus, dev_labels,
                                                                                             labels=labels)

labels = list(label2int.keys())

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
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)  # 对参数进行预测
    y_dev_predict = clf.predict(X_dev)  # 对参数进行预测

    # 获取结果报告
    print()
    print('The Test Accuracy of DT is:', clf.score(X_test, y_test))
    print(classification_report(y_test, y_predict, target_names=labels))
    print()
    print('The Dev Accuracy of DT is:', clf.score(X_dev, y_dev))
    print(classification_report(y_dev, y_dev_predict, target_names=labels))


def deep_search(maxdepth=500):
    depths = np.arange(1, maxdepth, step=2)
    training_scores = []
    testing_scores = []
    for depth in depths:
        clf = tree.DecisionTreeClassifier(max_depth=depth)
        clf = clf.fit(X_train, y_train)
        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)

        logging.info("{} deep, train score: {}, test score: {}".format(depth, train_score, test_score))
        training_scores.append(train_score)
        testing_scores.append(test_score)

    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(depths, training_scores, label="traing score", marker='o')
    ax.plot(depths, testing_scores, label="testing score", marker='*')
    ax.set_xlabel("maxdepth")
    ax.set_ylabel("score")
    ax.set_title("Decision Tree Classification")
    ax.legend(framealpha=0.5, loc='best')
    plt.show()


if __name__ == '__main__':
    deep_search()
