from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


def adaboost(training_data, training_labels, testing_data, testing_labels):
    classifier = AdaBoostClassifier()
    classifier.fit(training_data, training_labels)
    prediction = classifier.predict(testing_data)
    print(confusion_matrix(testing_labels, prediction))
    print("Accuracy:", accuracy_score(testing_labels, prediction))
