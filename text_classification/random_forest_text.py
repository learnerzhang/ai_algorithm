import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from matplotlib import pyplot as plt

from data.news_classification import create_vocabulary, read_news_corpus

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

vocab2int, int2vocab, label2int, int2label = create_vocabulary()
train_corpus, train_labels, test_corpus, test_labels, dev_corpus, dev_labels = read_news_corpus()

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


# 对参数 n 进行寻参，这里的参数范围是根据实际情况定义的
n_estimators_options = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
best_n_estimators = 0
best_acc = 0

# 寻参
training_scores = []
testing_scores = []
for n_estimators_size in n_estimators_options:
    alg = RandomForestClassifier(n_jobs=-1, n_estimators=n_estimators_size, max_depth=200)
    alg.fit(X_train, y_train)
    train_score = alg.score(X_train, y_train)
    test_score = alg.score(X_test, y_test)
    logging.info("{} estimators_size, train score: {}, test score: {}".format(n_estimators_size, train_score, test_score))
    training_scores.append(train_score)
    testing_scores.append(test_score)

    predict = alg.predict(X_test)
    acc = (y_test == predict).mean()
    # 更新最优参数和 acc
    if acc >= best_acc:
        best_acc = acc
        best_n_estimators = n_estimators_size
    print('[n_estimators, acc]:', n_estimators_size, acc)

# 绘图
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(n_estimators_options, training_scores, label="traing score", marker='o')
ax.plot(n_estimators_options, testing_scores, label="testing score", marker='*')
ax.set_xlabel("maxdepth")
ax.set_ylabel("score")
ax.set_title("Random Forest Classification")
ax.legend(framealpha=0.5, loc='best')
plt.show()

# 用最优参数进行训练
# rf = RandomForestClassifier(n_jobs=-1, n_estimators=best_n_estimators)
