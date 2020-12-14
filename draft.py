from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from requests_html import HTMLSession
from nltk.corpus import stopwords
from sklearn import svm
from time import sleep
from time import time
import pandas as pd
import nltk
import csv
import re


def scrape(url, name, cnt):
    cnt += 1
    if cnt >= 150:
        return
    # print(f"run page: {cnt}")
    hs = HTMLSession()

    try:
        url = url.replace("dp", "product-reviews")
    except Exception as e:
        print(e)
        quit()

    r = hs.get(url=url, headers={
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'accept-encoding': 'gzip, deflate, br',
        'accept-language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7,zh-TW;q=0.6',
        'sec-fetch-dest': 'document',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-site': 'none',
        'sec-fetch-user': '?1',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.75 Safari/537.36'})

    comments = r.html.find('div.a-section.review.aok-relative')

    fw = open(f'{name}.csv', 'a', encoding='utf8')  # output file
    writer = csv.writer(fw, lineterminator='\n')

    for a in comments:

        comment, star = 'NA', 'NA'  # initialize critic and text

        commentChunk = a.find('span.a-size-base.review-text.review-text-content > span')
        if commentChunk:
            comment = commentChunk[0].text.strip()

        starChunk = a.find('i > span.a-icon-alt')
        if starChunk: star = starChunk[0].text.strip()

        # star = a.find('i > span.a-icon-alt')[0].text
        # comment = a.find('span.a-size-base.review-text.review-text-content > span')[0].text

        writer.writerow([comment, star])

    fw.close()
    sleep(.75)
    pagination(r, name, cnt)

    r.close()


def pagination(attempt, name, cnt):
    next_page = attempt.html.find('li.a-last > a')
    if next_page:
        new_url = ''.join(next_page[0].absolute_links)
        # print(new_url)
        scrape(new_url, name, cnt)


def loadData(file):
    """
    read the reviews and their polarities from a given file
    while rate is more than 3 in 5, it would be a good review
    :param file: the path of review file
    :return: reviews and labels
    """
    reviews, labels = [], []
    f = pd.read_csv(file, header=None)
    for i in range(len(f)):
        reviews.append(f[0][i].replace('\n', '').lower())
        # f[1][i] --> rating
        rate = f[1][i].strip()
        rate = int(rate[0])
        if rate >= 3:
            labels.append(1)
        else:
            labels.append(0)
    return reviews, labels


def Filter(reviews):
    """
    decrease the dimension of dataset
    :param reviews: reviews from dataset
    :return: reviews without stop words
    """
    ans = []
    for review in reviews:
        temp = []
        review = re.sub(r'[^\w\s]', ' ', review)
        review = re.sub('[^a-z]', ' ', review)  # replace all non-letter characters

        ps = nltk.stem.porter.PorterStemmer()

        new_review = []
        for word in review.split():
            word = ps.stem(word)
            if word == '':
                continue  # ignore empty words and stopwords
            else:
                new_review.append(word)
        temp.append(' '.join(new_review))
        ans += temp
    return ans


def vt(predictors, counts_test, counts_train, lab_train):
    """
    Voting Classifier with different classification algorithms
    :param predictors: different classification algorithms
    :param counts_test: the transformed testing data
    :return: the accuracy score
    """
    VT = VotingClassifier(predictors)
    VT.fit(counts_train, lab_train)
    predicted = VT.predict(counts_test)
    return accuracy_score(predicted, lab_test)


def lgr_classifier(counts_train, lab_train):
    clf = LogisticRegression(solver='liblinear')
    LGR_grid = [{'penalty': ['l1', 'l2'], 'C': [0.5, 1, 1.5, 2, 3, 5, 10]}]
    gridsearchLGR = GridSearchCV(clf, LGR_grid, cv=5)
    return gridsearchLGR.fit(counts_train, lab_train)


def rf_classifier(counts_train, lab_train):
    clf = RandomForestClassifier(random_state=150, max_depth=600, min_samples_split=160)
    RF_grid = [{'n_estimators': [50, 100, 150, 200, 300, 500, 800, 1200, 1600, 2100],
                'criterion': ['gini', 'entropy'], 'max_features': ['auto', 'sqrt', 'log2']}]
    gridsearchRF = GridSearchCV(clf, RF_grid, cv=5)
    return gridsearchRF.fit(counts_train, lab_train)


def knn_classifier(counts_train, lab_train):
    clf = KNeighborsClassifier()
    KNN_grid = [{'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17],
                 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'brute']}]
    gridsearchKNN = GridSearchCV(clf, KNN_grid, cv=5)
    return gridsearchKNN.fit(counts_train, lab_train)


def dt_classifier(counts_train, lab_train):
    clf = DecisionTreeClassifier()
    DT_grid = [{'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 'criterion': ['gini', 'entropy'],
                'splitter': ['best', 'random']}]
    gridsearchDT = GridSearchCV(clf, DT_grid, cv=5)
    return gridsearchDT.fit(counts_train, lab_train)


def nb_classifier(counts_train, lab_train):
    clf = MultinomialNB()
    NB_grid = [{'alpha': [0.0001, 0.001, 0.01, 0.1, 0.8, 1, 10], 'fit_prior': [True, False]}]
    gridsearchNB = GridSearchCV(clf, NB_grid, cv=5)
    return gridsearchNB.fit(counts_train, lab_train)


def svm_classifier(counts_train, lab_train):
    clf = svm.SVC()
    SVM_grid = [{'C': [0.0001, 0.001, 0.01, 0.1, 0.8, 1, 10], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}]
    gridsearchSVM = GridSearchCV(clf, SVM_grid, cv=5)
    return gridsearchSVM.fit(counts_train, lab_train)


if __name__ == "__main__":
    all_start = time()

    cnt_train, cnt_test = 0, 0
    print('start scraping...')
    start = time()
    visited = set()
    scrape('https://www.amazon.com/Sennheiser-Momentum-Cancelling-Headphones-Functionality/dp/B07VW98ZKG', name='test', cnt=cnt_test)
    print("test set is ready...")
    scrape('https://www.amazon.com/dp/B07HKHWX7X/', name='train', cnt=cnt_train)
    print("train set is ready...")
    print('scraping finished')
    print(f"scrape time: {time() - start}s")  # 212.496s

    start = time()
    print('start training...')

    rev_train, lab_train = loadData('D:\\sunjiayi\\Stevens Studying File\\629 Internet of Things\\AmazonCommentsAnalysis\\train.csv')
    rev_test, lab_test = loadData('D:\\sunjiayi\\Stevens Studying File\\629 Internet of Things\\AmazonCommentsAnalysis\\test.csv')
    # remove the noise
    rev_train = Filter(rev_train)
    rev_test = Filter(rev_test)

    # Build a counter based on the training dataset
    counter = CountVectorizer(stop_words=stopwords.words('english'))
    counter.fit(rev_train)

    # count the number of times each term appears in a document and transform each doc into a count vector
    counts_train = counter.transform(rev_train)  # transform the training data
    counts_test = counter.transform(rev_test)  # transform the testing data

    # fit the models
    lgr_time = time()
    lgr_classifier(counts_train, lab_train)
    print(f"Logistic regression finished, run time: {time() - lgr_time}s")  # 2.1269s

    rf_time = time()
    rf_classifier(counts_train, lab_train)
    print(f"Random Forest finished, run time: {time() - rf_time}s")  # 819.8551s

    knn_time = time()
    knn_classifier(counts_train, lab_train)
    print(f"KNN finished, run time: {time() - knn_time}s")  # 36.7737s

    dt_time = time()
    dt_classifier(counts_train, lab_train)
    print(f"Decision tree finished, run time: {time() - dt_time}s")  # 5.9181s

    nb_time = time()
    nb_classifier(counts_train, lab_train)
    print(f"Naive Bayes finished, run time: {time() - nb_time}s")  # 0.2991s

    svm_time = time()
    svm_classifier(counts_train, lab_train)
    print(f"SVM finished, run time: {time() - svm_time}s")  # 274.7579s

    predictors = [('lreg', LogisticRegression()), ('rf', RandomForestClassifier()), ('knn', KNeighborsClassifier()),
                  ('dt', DecisionTreeClassifier()), ('nb', MultinomialNB()), ('svm', svm.SVC())]

    score = vt(predictors, counts_test, counts_train, lab_train)

    print(f"all finished, run time: {time() - start}s")  # 1150.1382s
    print(f"accuracy: {score * 100}%")  # 80.20304568527918%

    print(f"all finished, run time: {time() - all_start}s")  # 1362.636911869049s
