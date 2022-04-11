from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (
    plot_roc_curve,
    accuracy_score,
    plot_confusion_matrix,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from cross_validation import tune_number_of_decision_stumps, tune_k_neighbours
import time
import numpy
from utils import time_it
from results import plot_confusion_matrix, plot_roc_curve


class Classification:
    def __init__(self, training_data, training_labels, testing_data, testing_labels):
        self.training_data = training_data
        self.training_labels = training_labels
        self.testing_data = testing_data
        self.testing_labels = testing_labels

    def compare_classifiers(self):
        """Compare results of the three classifiers; adaboost, kNN,"""
        self.adaboost()
        self.kNN()

    def adaboost(self):
        """Adaboost classifier using cross-validation to determine the number of decision stumps"""
        decision_stumps_lowest_error = tune_number_of_decision_stumps(self.training_data)
        classifier = AdaBoostClassifier(n_estimators=decision_stumps_lowest_error)
        self._classify(classifier)

    def kNN(self):
        """K Nearest Neighbour classifier, using cross-validation to determine k, the number of nearest neighbours"""
        k_value_lowest_error = tune_k_neighbours(self.training_data)
        classifier = KNeighborsClassifier(n_neighbors=k_value_lowest_error)
        self._classify(classifier)

    def _classify(self, classifier):
        classifier.fit(self.training_data, self.training_labels)
        prediction = classifier.predict(self.testing_data)
        self._plot_results(classifier, prediction)

    def _error_rate(self, prediction):
        """Calculate error rate using classifier prediction and test labels"""
        return 1 - accuracy_score(self.testing_labels, prediction)

    def _plot_results(self, classifier, prediction):
        """Generate confusion matrix and plot ROC curve to show prediction results"""
        print(f"{classifier} Error Rate: = {self._error_rate(prediction)}")
        plot_confusion_matrix(classifier, self.testing_labels, prediction)
        plot_roc_curve(self.testing_labels, prediction)


def adaboost(training_data, training_labels, testing_data, testing_labels):
    """Build an Adaboost classifier using cross-validation to determine the number of decision stumps"""
    # num_of_decision_stumps = tune_number_of_decision_stumps(training_data)
    classifier = AdaBoostClassifier()
    classifier.fit(training_data, training_labels)
    prediction = classifier.predict(testing_data)
    print(f"AdaBoost Error Rate = {1 - accuracy_score(testing_labels, prediction)}")
    plot_confusion_matrix(classifier, testing_labels, prediction)
    plot_roc_curve(testing_labels, prediction)


# precdiction on left -> actual on top
# summary of errors on a per class basis = confusion matrix
# true positive (correct)     false positive (error)
# false negative (error)    true negatives (correct)
def kNN(k_values, training_data, training_labels, testing_data, testing_labels):
    for k in k_values:
        start = time.time()
        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(training_data, training_labels)
        end = time.time()
        prediction = classifier.predict(testing_data)
        print(f"Computational time for training KNN: {end - start}")
        # print(f"Confusion Matrix: \n {confusion_matrix(testing_labels, prediction)}")
        print(f"KNN (k={k}) Error Rate = {1 - accuracy_score(testing_labels, prediction)}")
