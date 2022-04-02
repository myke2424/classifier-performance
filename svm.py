from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score


def svmM(training_data, training_labels, testing_data, testing_labels):
    classifier = svm.SVC(kernel='linear')
    classifier.fit(training_data, training_labels)
    prediction = classifier.predict(testing_data)
    print(confusion_matrix(testing_labels, prediction))
    print("Accuracy:", accuracy_score(testing_labels, prediction))



