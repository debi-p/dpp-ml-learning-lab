import numpy as np


def accuracy_score(predicted, actual):
    if len(actual) == 0:
        return 0.0
    return float(np.mean(predicted == actual))


def confusion_matrix(predicted, actual, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for actual_class, predicted_class in zip(actual, predicted):
        matrix[actual_class, predicted_class] += 1
    return matrix
