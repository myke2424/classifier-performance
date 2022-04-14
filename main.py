from pathlib import Path

import numpy

from classifiers import Classification

TRAINING_DATA_FILE_PATH = Path("data/Training.csv")
TESTING_DATA_FILE_PATH = Path("data/Testing.csv")
LABEL_INDEX = 11

training_data = numpy.loadtxt(TRAINING_DATA_FILE_PATH, delimiter=",", skiprows=1)
training_data_without_labels = training_data[:, :LABEL_INDEX]
training_labels = training_data[:, LABEL_INDEX]

testing_data = numpy.loadtxt(TESTING_DATA_FILE_PATH, delimiter=",", skiprows=1)
testing_data_without_labels = testing_data[:, :LABEL_INDEX]
testing_labels = testing_data[:, LABEL_INDEX]


def main():
    classification = Classification(
        training_data=training_data,
        testing_data=testing_data,
        training_labels=training_labels,
        testing_labels=testing_labels,
    )
    classification.compare_classifiers()


if __name__ == "__main__":
    main()
