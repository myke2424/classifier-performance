from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix, accuracy_score


def kNN(training_data, training_labels, testing_data, testing_labels):
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(training_data, training_labels)
    prediction = classifier.predict(testing_data)
    print(confusion_matrix(testing_labels, prediction))
    print("Accuracy:", accuracy_score(testing_labels, prediction))
