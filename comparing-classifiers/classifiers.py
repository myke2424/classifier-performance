import numpy as np
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, plot_confusion_matrix, plot_roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from results import plot_confusion_matrix, plot_roc_curve


class Classification:
    def __init__(self, training_data, training_labels, testing_data, testing_labels):
        self.training_data = training_data
        self.training_labels = training_labels
        self.testing_data = testing_data
        self.testing_labels = testing_labels

    def compare_classifiers(self):
        """Compare results of the three classifiers; adaboost, kNN and SVM"""
        self.adaboost()
        self.kNN()
        self.svm_()

    def adaboost(self):
        """Adaboost classifier using cross-validation to determine the number of decision stumps"""
        # decision_stumps_lowest_error = tune_number_of_decision_stumps(self.training_data)
        param_grid = {"n_estimators": np.arange(1, 50)}
        total_decision_stumps = self._hypertune_param(AdaBoostClassifier, param_grid).get("n_estimators")
        classifier = AdaBoostClassifier(n_estimators=total_decision_stumps)
        self._classify(classifier)

    def kNN(self):
        """K Nearest Neighbour classifier, using cross-validation to determine k, the number of nearest neighbours"""
        param_grid = {"n_neighbors": np.arange(1, 50)}
        k_value = self._hypertune_param(KNeighborsClassifier, param_grid).get("n_neighbors")
        classifier = KNeighborsClassifier(n_neighbors=k_value)
        self._classify(classifier)

    def svm_(self):
        """SVM classifier, using cross-validation to determine kernel type and kernel coefficient"""
        param_grid = [
            {"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
            {"kernel": ["linear"], "C": [1, 10, 100, 1000]},
        ]
        params = self._hypertune_param(svm.SVC, param_grid)
        classifier = svm.SVC(**params)
        self._classify(classifier)

    def _classify(self, classifier):
        classifier.fit(self.training_data, self.training_labels)
        prediction = classifier.predict(self.testing_data)
        self._plot_results(classifier, prediction)

    def _hypertune_param(self, classifier, param_grid, folds=5):
        """
        Use cross-validation to hypertune parameters for our classifiers, using 5-fold for partitioning the data into
        training and validation sets, and grid-search to find the optimal value for all params in the param grid
        """
        cv_classifier = GridSearchCV(classifier(), param_grid, cv=folds)
        cv_classifier.fit(self.training_data, self.training_labels)
        print(f"Grid search results: {cv_classifier.best_params_}")
        return cv_classifier.best_params_

    def _error_rate(self, prediction):
        """Calculate error rate using classifier prediction and test labels"""
        return 1 - accuracy_score(self.testing_labels, prediction)

    def _plot_results(self, classifier, prediction):
        """Generate confusion matrix and plot ROC curve to show prediction results"""
        print(f"{classifier} Error Rate: = {self._error_rate(prediction)}")
        plot_confusion_matrix(classifier, self.testing_labels, prediction)
        plot_roc_curve(classifier, self.testing_labels, prediction)
