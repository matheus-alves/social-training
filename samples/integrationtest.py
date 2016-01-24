from functions import *

# Constants
UNLABELED_RATE = UnlabeledDataRates.eighty

CLASS_THRESHOLD = 0.65

POSITIVE_CLASS_LABEL = 0
NEGATIVE_CLASS_LABEL = 1

import os
import numpy as np

print('Loading Data Set')

CWD = os.getcwd()

# Loads the CSV file as a numpy matrix
raw_data = np.loadtxt(CWD + '/datasets/diabetes.txt', delimiter=",")

# Creates the instances and labels sets
instances = raw_data[:, 0:8]
labels = raw_data[:, 8]

# Creates tha data set abstraction class
data_set = DataSet(instances, labels, UNLABELED_RATE)

print('\nPositive class ratio:', data_set.class_ratio[POSITIVE_CLASS_LABEL])
print('Negative class ratio:', data_set.class_ratio[NEGATIVE_CLASS_LABEL])

print('\nTest data size:', len(data_set.test_data.instances))
print('Train data size:', len(data_set.unlabeled_data.instances) +
      len(data_set.labeled_data.instances))

print('\nLabeled data size:', len(data_set.labeled_data.instances))
print('Unlabeled data size:', len(data_set.unlabeled_data.instances))

#Classification
social_training = SocialTraining()

social_training.set_binary_classification_parameters(
    class_threshold=CLASS_THRESHOLD)

social_training.set_classifiers([
    ClassifierTypes.linear_discriminant_analysis,
    ClassifierTypes.gradient_boosting,
    ClassifierTypes.logistic_regression,
    ClassifierTypes.k_nearest_neighbors,
    ClassifierTypes.gaussian_naive_bayes,
    ClassifierTypes.quadratic_discriminant_analysis,
    ClassifierTypes.bernoulli_naive_bayes
    ])

social_training.apply_social_training(data_set)


# TODO change classifiers to use cross-validation
# TODO add more classifiers diversity
# TODO compare with tri-training results