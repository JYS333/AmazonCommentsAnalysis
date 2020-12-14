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
from time import time, sleep
from sklearn import svm
import pandas as pd
import nltk
import csv
import re


def scrape(url):
    """
    Scrape comments from Amazon.com, write them(comments and rating) into a csv file
    :param url: the link from Amazon
    :return: None
    """
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

    fw = open('reviews.csv', 'a', encoding='utf8')  # output file
    writer = csv.writer(fw, lineterminator='\n')

    for a in comments:

        comment, star = 'NA', 'NA'  # initialize critic and text

        commentChunk = a.find('span.a-size-base.review-text.review-text-content > span')
        if commentChunk: comment = commentChunk[0].text.strip()

        starChunk = a.find('i > span.a-icon-alt')
        if starChunk: star = starChunk[0].text.strip()

        # star = a.find('i > span.a-icon-alt')[0].text
        # comment = a.find('span.a-size-base.review-text.review-text-content > span')[0].text

        writer.writerow([comment, star])

    fw.close()
    sleep(.75)
    pagination(r)

    r.close()


def pagination(attempt):
    """
    find a new page, be called in scrape()
    :param attempt: Amazon link in scrape()
    :return:None
    """
    next_page = attempt.html.find('li.a-last > a')
    if next_page:
        new_url = ''.join(next_page[0].absolute_links)
        # print(new_url)
        scrape(new_url)


def loadData(file):
    """
    read the reviews and their polarities from a given file
    :param file: the path of review file
    :return: reviews and labels
    """
    reviews, labels = [], []
    f = pd.read_csv(file, header=None)
    for i in range(len(f)):
        reviews.append(f[0][i].replace('\n', ''))
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
        # review = re.sub(r'[^\w\s]', ' ', review)
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
    :param counts_test: the transformed testing reviews
    :param counts_train: the transformed training reviews
    :param lab_train: the rating of training comments
    :return: the accuracy score
    """
    VT = VotingClassifier(predictors)
    VT.fit(counts_train, lab_train)
    predicted = VT.predict(counts_test)
    return accuracy_score(predicted, lab_test)


def lgr_classifier(counts_train, lab_train):
    """
    Logistic Regression Classifier with grid search
    :param counts_train: the transformed training reviews
    :param lab_train: the rating of training comments
    :return: Accuracy of Logistic regression classifier
    """
    clf = LogisticRegression(solver='liblinear')
    LGR_grid = [{'penalty': ['l1', 'l2'], 'C': [0.5, 1, 1.5, 2, 3, 5, 10]}]
    gridsearchLGR = GridSearchCV(clf, LGR_grid, cv=5)
    # LGR_fit, LGR_score = gridsearchLGR.fit(counts_train, lab_train), gridsearchLGR.score(counts_train)
    return gridsearchLGR.fit(counts_train, lab_train)


def rf_classifier(counts_train, lab_train):
    """
    Random Forest Classifier with grid search
    :param counts_train: the transformed training reviews
    :param lab_train: the rating of training comments
    :return: Accuracy of Random Forest Classifier
    """
    clf = RandomForestClassifier(random_state=150, max_depth=600, min_samples_split=160)
    RF_grid = [{'n_estimators': [50, 100, 150, 200, 300, 500, 800, 1200, 1600, 2100],
                'criterion': ['gini', 'entropy'], 'max_features': ['auto', 'sqrt', 'log2']}]
    gridsearchRF = GridSearchCV(clf, RF_grid, cv=5)
    return gridsearchRF.fit(counts_train, lab_train)


def knn_classifier(counts_train, lab_train):
    """
    K-Nearest Neighbors Classifier with grid search
    :param counts_train: the transformed training reviews
    :param lab_train: the rating of training comments
    :return: Accuracy of K-Nearest Neighbors Classifier
    """
    clf = KNeighborsClassifier()
    KNN_grid = [{'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17],
                 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'brute']}]
    gridsearchKNN = GridSearchCV(clf, KNN_grid, cv=5)
    return gridsearchKNN.fit(counts_train, lab_train)


def dt_classifier(counts_train, lab_train):
    """
    Decision Tree Classifier with grid search
    :param counts_train: the transformed training reviews
    :param lab_train: the rating of training comments
    :return: Accuracy of Decision Tree Classifier
    """
    clf = DecisionTreeClassifier()
    DT_grid = [{'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 'criterion': ['gini', 'entropy'],
                'splitter': ['best', 'random']}]
    gridsearchDT = GridSearchCV(clf, DT_grid, cv=5)
    return gridsearchDT.fit(counts_train, lab_train)


def nb_classifier(counts_train, lab_train):
    """
    Naive Bayes Classifier with grid search
    :param counts_train: the transformed training reviews
    :param lab_train: the rating of training comments
    :return: Accuracy of Naive Bayes Classifier
    """
    clf = MultinomialNB()
    NB_grid = [{'alpha': [0.0001, 0.001, 0.01, 0.1, 0.8, 1, 10], 'fit_prior': [True, False]}]
    gridsearchNB = GridSearchCV(clf, NB_grid, cv=5)
    return gridsearchNB.fit(counts_train, lab_train)


def svm_classifier(counts_train, lab_train):
    """
    Support Vector Machine Classifier with grid search
    :param counts_train: the transformed training reviews
    :param lab_train: the rating of training comments
    :return: Accuracy of Support Vector Machine Classifier
    """
    clf = svm.SVC()
    SVM_grid = [{'C': [0.0001, 0.001, 0.01, 0.1, 0.8, 1, 10], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}]
    gridsearchSVM = GridSearchCV(clf, SVM_grid, cv=5)
    return gridsearchSVM.fit(counts_train, lab_train)


if __name__ == "__main__":
    # initial the start time
    all_start = time()

    print('start scraping...')
    start = time()  # initial the scrape time
    # Scrape comments from Amazon.com, write them(comments and rating) into a csv file
    scrape('https://www.amazon.com/Sennheiser-Momentum-Cancelling-Headphones-Functionality/dp/B07VW98ZKG')
    print('scraping finished')
    print(f"scrape time: {time() - start}")

    start = time()  # initial the training time
    print('start training...')

    # load the training set(reviews and labels)
    rev_train, lab_train = loadData('D:\\sunjiayi\\Stevens_Studying_File\\629\\AmazonCommentsAnalysis\\train.csv')
    # load the testing set(rewiews and labels)
    rev_test, lab_test = loadData('D:\\sunjiayi\\Stevens_Studying_File\\629\\AmazonCommentsAnalysis\\test.csv')

    # Correct wrong expressions
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
    print(f"Logistic regression finished, run time: {time() - lgr_time}")

    rf_time = time()
    rf_classifier(counts_train, lab_train)
    print(f"Random Forest finished, run time: {time() - rf_time}")

    knn_time = time()
    knn_classifier(counts_train, lab_train)
    print(f"KNN finished, run time: {time() - knn_time}")

    dt_time = time()
    dt_classifier(counts_train, lab_train)
    print(f"Decision tree finished, run time: {time() - dt_time}")

    nb_time = time()
    nb_classifier(counts_train, lab_train)
    print(f"Naive Bayes finished, run time: {time() - nb_time}")

    svm_time = time()
    svm_classifier(counts_train, lab_train)
    print(f"SVM finished, run time: {time() - svm_time}")

    # vote
    predictors = [('lreg', LogisticRegression()), ('rf', RandomForestClassifier()), ('knn', KNeighborsClassifier()),
                  ('dt', DecisionTreeClassifier()), ('nb', MultinomialNB()), ('svm', svm.SVC())]

    score = vt(predictors, counts_test, counts_train, lab_train)

    print(f"all finished, run time: {time() - start}")  # 949.3523089885712
    print(f"accuracy: {score}")  # 0.8833333333333333

    print(f"all finished, run time: {time() - all_start}")
