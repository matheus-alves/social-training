NUM_OF_ATTRIBUTES = 6

CLASS_THRESHOLD = 0.42

POSITIVE_CLASS_LABEL = 1
NEGATIVE_CLASS_LABEL = 2

HEADERS = list()
HEADERS.append('@relation bupa')
HEADERS.append('@attribute mcv real')
HEADERS.append('@attribute alkphos real')
HEADERS.append('@attribute sgpt real')
HEADERS.append('@attribute sgot real')
HEADERS.append('@attribute gammagt real')
HEADERS.append('@attribute drinks real')
HEADERS.append('@attribute selector {1,2}')
HEADERS.append('@data')

print('Loading Data Set: ' + "bupa")

import os
import numpy as np

CWD = os.getcwd()

# Loads the CSV file as a numpy matrix
#raw_data = np.loadtxt(CWD + '/datasets/bupa.txt', delimiter=",")

from scipy.io.arff import loadarff
raw_data = loadarff(open(CWD + '/datasets/bupa.arff', 'r'))[0]

# Creates the instances and labels sets
instances = list()
labels = list()
for i in range(0, len(raw_data)):
    instance = list()
    for j in range(0, NUM_OF_ATTRIBUTES):
        instance.append(raw_data[i][j])

    instances.append(instance)
    labels.append(int(raw_data[i][NUM_OF_ATTRIBUTES]))

# Creates the instances and labels sets
# instances = raw_data[:, 0:NUM_OF_ATTRIBUTES ]
# labels = raw_data[:, NUM_OF_ATTRIBUTES ]