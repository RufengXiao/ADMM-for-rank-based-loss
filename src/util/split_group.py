import numpy as np

def train_test_split_group(X, y, group,test_size=0.2, random_state=None):
    n_samples, n_features = X.shape

    if random_state is not None:
        np.random.seed(random_state)

    shuffled_index = np.random.permutation(n_samples)

    n_test_samples = int(n_samples * test_size)

    test_index = shuffled_index[:n_test_samples]

    train_index = shuffled_index[n_test_samples:]

    X_train = X[train_index]
    X_test = X[test_index]
    y_train = y[train_index]
    y_test = y[test_index]
    group_train = group[train_index]
    group_test = group[test_index]

    return X_train, X_test, y_train, y_test, group_train, group_test