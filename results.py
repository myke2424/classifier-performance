from sklearn.metrics import confusion_matrix, RocCurveDisplay, roc_curve, auc, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def plot_confusion_matrix(classifier, testing_labels, prediction):
    cm = confusion_matrix(testing_labels, prediction, labels=classifier.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    disp.plot()
    plt.title(f"{classifier} Confusion Matrix")
    plt.show()


def plot_roc_curve(classifier, testing_labels, prediction):
    fpr, tpr, threshold = roc_curve(testing_labels, prediction)
    roc_auc = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    display.plot()
    plt.title(f"{classifier} ROC Curve")
    plt.show()
