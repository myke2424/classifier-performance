from classifiers import adaboost, kNN
import numpy
from sklearn.model_selection import KFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

TRAINING_DATA_FILE_PATH = "data/Training.csv"
TESTING_DATA_FILE_PATH = "data/Testing.csv"

training_data = numpy.loadtxt("data/Training.csv", delimiter=",", skiprows=1)
training_data_without_labels = training_data[:, :11]
training_labels = training_data[:, 11]

testing_data = numpy.loadtxt("data/Testing.csv", delimiter=",", skiprows=1)
testing_data_without_labels = testing_data[:, :11]
testing_labels = testing_data[:, 11]

kf3 = KFold(n_splits=3, shuffle=True)


def main():
    adaboost(
        training_data=training_data,
        training_labels=training_labels,
        testing_data=testing_data,
        testing_labels=testing_labels,
    )

    kNN(
        k_values=[3, 5, 8, 9, 10, 11, 12, 13, 14],
        training_data=training_data,
        training_labels=training_labels,
        testing_data=testing_data,
        testing_labels=testing_labels,
    )


if __name__ == "__main__":
    main()
