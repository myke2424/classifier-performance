import numpy

from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


def tune_number_of_decision_stumps(data: list, folds: int = 3, start: int = 1, stop: int = 50, step: int = 1) -> int:
    """Using k-fold cross validation, find the number of decision stumps that produce the lowest error rate"""

    k_fold = KFold(n_splits=folds, shuffle=True)

    average_errors = {}
    for i in range(start, stop, step):
        errors = []
        for train_index, validate_index in k_fold.split(data):
            train = numpy.take(data, train_index, axis=0)
            validate = numpy.take(data, validate_index, axis=0)
            classifier = AdaBoostClassifier(n_estimators=i)
            classifier.fit(train[:, :11], train[:, 11])

            prediction = classifier.predict(validate[:, :11])
            error_rate = 1 - accuracy_score(validate[:, 11], prediction)
            errors.append(error_rate)
        average_error = sum(errors) / len(errors)
        average_errors[i] = average_error
    param_with_lowest_error_rate = min(average_errors, key=average_errors.get)
    print(
        f"Using cross-validation, the number of decision stumps with the lowest error rate is: "
        f"{param_with_lowest_error_rate} \n "
        f" Error rate: {average_errors[param_with_lowest_error_rate]}"
    )
    return param_with_lowest_error_rate


def tune_k_neighbours(data: list, folds: int = 3, start: int = 3, stop: int = 100, step: int = 3) -> int:
    """Using k-fold cross validation, find k, the number of nearest neighbors, that produces the lowest error rate"""

    k_fold = KFold(n_splits=folds, shuffle=True)

    average_errors = {}
    for k in range(start, stop, step):
        errors = []
        for train_index, validate_index in k_fold.split(data):
            train = numpy.take(data, train_index, axis=0)
            validate = numpy.take(data, validate_index, axis=0)
            classifier = KNeighborsClassifier(n_neighbors=k)
            classifier.fit(train[:, :11], train[:, 11])
            prediction = classifier.predict(validate[:, :11])
            error_rate = 1 - accuracy_score(validate[:, 11], prediction)
            errors.append(error_rate)
        average_error = sum(errors) / len(errors)
        average_errors[k] = average_error
    param_with_lowest_error_rate = min(average_errors, key=average_errors.get)
    print(
        f"Using cross-validation, the number of decision stumps with the lowest error rate is: "
        f"{param_with_lowest_error_rate} \n "
        f" Error rate: {average_errors[param_with_lowest_error_rate]}"
    )
    return param_with_lowest_error_rate
