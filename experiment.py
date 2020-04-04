from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.cluster import KMeans
import numpy as np
from sklearn.model_selection import cross_validate

from tools import extract_data, split_data, shuffle_data


def get_data(path, training_percent=0.7, percent_zeroes=0.5, n_nonzero_repeat=0, shuffle=True, normalize=False):
    X, y = extract_data(path, normalize)
    X_train, y_train, X_test, y_test = split_data(X, y, training_percent=training_percent, percent_zeroes=percent_zeroes, n_nonzero_repeat=n_nonzero_repeat)
    if shuffle:
        X_train, y_train = shuffle_data(X_train, y_train)
        X_test, y_test = shuffle_data(X_test, y_test)
    return X_train, y_train, X_test, y_test


def baseline(path, output=True):
    X_train, y_train, X_test, y_test = get_data(path)
    dc = DummyClassifier(strategy='most_frequent')
    dc.fit(X_train, y_train)
    if output:
        print('DummyClassifier score: {}'.format(dc.score(X_test, y_test)))


def neural_net(path, output=True, max_iter=200, cv=False):
    mlp = MLPClassifier((30,), momentum=0.9, activation='relu', max_iter=max_iter)
    if cv:
        X, y = extract_data(path, True)
        Results = cross_validate(mlp, X, y, cv=3, return_train_score=True)
        if output:
            print(Results['train_score'])
            print(Results['test_score'])
            print(np.average(Results['test_score']))
    else:
        X_train, y_train, X_test, y_test = get_data(path,percent_zeroes=.4,training_percent=.7,n_nonzero_repeat=1, normalize=True)
        mlp.fit(X_train, y_train)
        if output:
            print('MLPClassifer score: {}'.format(mlp.score(X_test, y_test)))
            a = mlp.predict(X_test)


def k_means(path, output=True, include_labels=True, n_clusters=13):
    X_train, y_train, X_test, y_test = get_data(path, normalize=True)
    neighbors = KMeans(n_clusters=n_clusters)
    if include_labels:
        neighbors.fit(np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1))
    else:
        neighbors.fit(X_train)
    if output:
        print('KMeans SSE: {}'.format(neighbors.inertia_))
        print('KMeans Score: {}'.format(neighbors.score(np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1))))


def main():
    data_path = './data/matthew_delta.csv'
    baseline(data_path)
    neural_net(data_path, max_iter=1000)
    #k_means(data_path)


if __name__ == '__main__':
    main()
