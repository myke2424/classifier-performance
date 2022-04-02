import numpy
from knn import kNN
from adaboost import adaboost
from svm import svmM

TRAINING_DATA_FILE_PATH = "data/training_data.txt"  # 900 data points, 4 attributes, 450 label = 0, 450 label = 1
TESTING_DATA_FILE_PATH = "data/test_data.txt"  # 320 data points, 4 attributes, 160 label = 0, 160 label = 1

training_data = numpy.loadtxt(TRAINING_DATA_FILE_PATH, delimiter=",", dtype="float")
training_data_without_labels = training_data[:, :4]
training_labels = training_data[:, 4]

testing_data = numpy.loadtxt(TESTING_DATA_FILE_PATH, delimiter=",", dtype="float")
testing_data_without_labels = testing_data[:, :4]
testing_labels = testing_data[:, 4]


def main():
    kNN(training_data=training_data_without_labels, training_labels=training_labels,
        testing_data=testing_data_without_labels, testing_labels=testing_labels)

    adaboost(training_data=training_data_without_labels, training_labels=training_labels,
             testing_data=testing_data_without_labels, testing_labels=testing_labels)

    svmM(training_data=training_data_without_labels, training_labels=training_labels,
        testing_data=testing_data_without_labels, testing_labels=testing_labels)


if __name__ == '__main__':
    main()
