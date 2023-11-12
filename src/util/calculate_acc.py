import numpy as np

def calculate_accuracy(w, X_test, y_test, threshold=0.5,loss='binary_cross_entropy'):
    if loss == 'binary_cross_entropy':
        predictions = np.dot(X_test, w)
        class_probs = np.zeros((predictions.shape))
        class_probs[predictions >= 0] = 1 / (1 + np.exp(-predictions[predictions >= 0]))
        class_probs[predictions < 0] = np.exp(predictions[predictions < 0]) / (np.exp(predictions[predictions < 0]) + 1)
        binary_preds = (class_probs >= threshold).astype(int)
        binary_preds[binary_preds == 0] = -1
        accuracy = np.mean(binary_preds == y_test)
    elif loss == 'hinge':
        predictions = np.dot(X_test, w)
        binary_preds = (predictions >= 0).astype(int)
        binary_preds[binary_preds == 0] = 1
        accuracy = np.mean(binary_preds == y_test)
    else:
        raise ValueError(
            f"loss '{loss}' is not supported! Options: ['binary_cross_entropy','hinge']")
    return accuracy