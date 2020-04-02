from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.cluster import KMeans
import numpy as np

from tools import extract_data, split_data, shuffle_data


def get_data(path, training_percent=0.7, percent_zeroes=0.5, shuffle=True):
    X, y = extract_data(path)
    X_train, y_train, X_test, y_test = split_data(X, y, training_percent=training_percent, percent_zeroes=percent_zeroes)
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


def neural_net(path, output=True, max_iter=200):
    X_train, y_train, X_test, y_test = get_data(path)
    mlp = MLPClassifier(max_iter=max_iter)
    mlp.fit(X_train, y_train)
    if output:
        print('MLPClassifer score: {}'.format(mlp.score(X_test, y_test)))


def k_means(path, output=True, n_clusters=8):
    X_train, y_train, X_test, y_test = get_data(path)
    neighbors = KMeans(n_clusters=n_clusters)
    neighbors.fit(np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1))
    if output:
        print('KMeans SSE: {}'.format(neighbors.inertia_))
        print('KMeans Score: {}'.format(neighbors.score(np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1))))


def main():
    data_path = './data/matthew_delta.csv'
    baseline(data_path)
    neural_net(data_path)
    k_means(data_path)


if __name__ == '__main__':
    main()
