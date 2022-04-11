from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, RocCurveDisplay, roc_curve, auc, plot_roc_curve, accuracy_score, plot_confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from cross_validation import tune_number_of_decision_stumps
import time
import numpy
import matplotlib.pyplot as plt
from utils import time_it


def adaboost(training_data, training_labels, testing_data, testing_labels):
    """Build an Adaboost classifier using cross-validation to determine the number of decision stumps"""
    # num_of_decision_stumps = tune_number_of_decision_stumps(training_data)
    classifier = AdaBoostClassifier()
    classifier.fit(training_data, training_labels)
    prediction = classifier.predict(testing_data)
    cmatrix = confusion_matrix(testing_labels, prediction, labels=classifier.classes_)
    print(cmatrix)
    disp = ConfusionMatrixDisplay(confusion_matrix=cmatrix, display_labels=classifier.classes_)
    disp.plot()
    plt.show()
    print(f"AdaBoost Error Rate = {1 - accuracy_score(testing_labels, prediction)}")
    fpr, tpr, threshold = roc_curve(testing_labels, prediction)
    roc_auc = auc(fpr, tpr) # Compute Area Under the Curve (AUC)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    display.plot()
    plt.show()
    # plt.title('Receiver Operating Characteristic')
    # plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    # plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.show()


# precdiction on left -> actual on top
# summary of errors on a per class basis = confusion matrix
# true positive (correct)     false positive (error)
# false negative (error)    true negatives (correct)
def kNN(k_values, training_data, training_labels, testing_data, testing_labels):
    for k in k_values:
        classifier = KNeighborsClassifier(n_neighbors=k)

        start = time.time()
        classifier.fit(training_data, training_labels)
        end = time.time()
        # print(f"Computational time for training KNN: {end - start}")

        prediction = classifier.predict(testing_data)
        # print(f"Confusion Matrix: \n {confusion_matrix(testing_labels, prediction)}")
        print(f"KNN (k={k}) Error Rate = {1 - accuracy_score(testing_labels, prediction)}")
