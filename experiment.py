from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.cluster import KMeans
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error

from tools import extract_data, split_data, shuffle_data


def get_data(path, training_percent=0.7, percent_zeroes=0.5, n_nonzero_repeat=0, shuffle=True, normalize=False):
    X, y = extract_data(path, normalize)
    X_train, y_train, X_test, y_test = split_data(X, y, training_percent=training_percent, percent_zeroes=percent_zeroes, n_nonzero_repeat=n_nonzero_repeat)
    if shuffle:
        X_train, y_train = shuffle_data(X_train, y_train)
        X_test, y_test = shuffle_data(X_test, y_test)
    return X_train, y_train, X_test, y_test


def baseline_classification(path, output=True, percent_zeroes=.5, training_percent=.7,n_nonzero_repeat=1):
    X_train, y_train, X_test, y_test = get_data(path, percent_zeroes=percent_zeroes, training_percent=training_percent,
                                                n_nonzero_repeat=n_nonzero_repeat, normalize=False)
    dc = DummyClassifier(strategy='most_frequent')
    dc.fit(X_train, y_train)
    if output:
        print('DummyClassifier score: {}'.format(dc.score(X_test, y_test)))

def baseline_regression(path, output=True, percent_zeroes=.5, training_percent=.7,n_nonzero_repeat=1):
    X_train, y_train, X_test, y_test = get_data(path, percent_zeroes=percent_zeroes, training_percent=training_percent,
                                                n_nonzero_repeat=n_nonzero_repeat, normalize=True)
    dc = DummyRegressor()
    dc.fit(X_train, y_train)
    if output:
        print('DummyRegressor score: {}'.format(meanSquaredError(dc, X_test, y_test)))


def neural_net_classification(path, output=True, max_iter=200, cv=False, percent_zeroes=.5, training_percent=.7,
                              n_nonzero_repeat=1):
    mlp = MLPClassifier((30,), momentum=0.9, activation='relu', max_iter=max_iter)
    if cv:
        X, y = extract_data(path, True)
        Results = cross_validate(mlp, X, y, cv=3, return_train_score=True)
        if output:
            print(Results['train_score'])
            print(Results['test_score'])
            print(np.average(Results['test_score']))
    else:
        X_train, y_train, X_test, y_test = get_data(path,percent_zeroes=percent_zeroes,training_percent=training_percent,
                                                    n_nonzero_repeat=n_nonzero_repeat, normalize=False)
        mlp.fit(X_train, y_train)
        if output:
            print('MLPClassifer score: {}'.format(mlp.score(X_test, y_test)))
            a = mlp.predict(X_test)

def neural_net_regression(path, output=True, max_iter=200, percent_zeroes=.5, training_percent=.7, n_nonzero_repeat=1):
    mlp = MLPRegressor(hidden_layer_sizes=(30,), momentum=0.9, activation='relu', max_iter=max_iter)
    X_train, y_train, X_test, y_test = get_data(path, percent_zeroes=percent_zeroes, training_percent=training_percent,
                                                n_nonzero_repeat=n_nonzero_repeat, normalize=True)
    mlp.fit(X_train, y_train)
    if output:
        print('MLPRegressor score: {}'.format(meanSquaredError(mlp, X_test, y_test)))

def meanSquaredError(classifier, X_test, y_test):
    y_pred = classifier.predict(X_test)
    return mean_squared_error(y_test, y_pred)


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
    baseline_classification(data_path, training_percent=0.7, percent_zeroes=0.4, n_nonzero_repeat=1)
    neural_net_classification(data_path, max_iter=1000, training_percent=0.7,percent_zeroes=0.4,n_nonzero_repeat=1)
    baseline_regression(data_path, training_percent=0.7, percent_zeroes=0.4, n_nonzero_repeat=1)
    neural_net_regression(data_path, max_iter=1000, training_percent=0.7,percent_zeroes=0.4,n_nonzero_repeat=1)
    #k_means(data_path)


if __name__ == '__main__':
    main()
