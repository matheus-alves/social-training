NUM_OF_ATTRIBUTES = 34

CLASS_THRESHOLD = 0.35

POSITIVE_CLASS_LABEL = 0
NEGATIVE_CLASS_LABEL = 1

print('Loading Data Set: ' + "ionosphere")

import os
import numpy as np

HEADERS = list()
HEADERS.append("@relation ionosphere")
HEADERS.append("@attribute a01 real")
HEADERS.append("@attribute a02 real")
HEADERS.append("@attribute a03 real")
HEADERS.append("@attribute a04 real")
HEADERS.append("@attribute a05 real")
HEADERS.append("@attribute a06 real")
HEADERS.append("@attribute a07 real")
HEADERS.append("@attribute a08 real")
HEADERS.append("@attribute a09 real")
HEADERS.append("@attribute a10 real")
HEADERS.append("@attribute a11 real")
HEADERS.append("@attribute a12 real")
HEADERS.append("@attribute a13 real")
HEADERS.append("@attribute a14 real")
HEADERS.append("@attribute a15 real")
HEADERS.append("@attribute a16 real")
HEADERS.append("@attribute a17 real")
HEADERS.append("@attribute a18 real")
HEADERS.append("@attribute a19 real")
HEADERS.append("@attribute a20 real")
HEADERS.append("@attribute a21 real")
HEADERS.append("@attribute a22 real")
HEADERS.append("@attribute a23 real")
HEADERS.append("@attribute a24 real")
HEADERS.append("@attribute a25 real")
HEADERS.append("@attribute a26 real")
HEADERS.append("@attribute a27 real")
HEADERS.append("@attribute a28 real")
HEADERS.append("@attribute a29 real")
HEADERS.append("@attribute a30 real")
HEADERS.append("@attribute a31 real")
HEADERS.append("@attribute a32 real")
HEADERS.append("@attribute a33 real")
HEADERS.append("@attribute a34 real")
HEADERS.append("@attribute class {0, 1}")
HEADERS.append("@data")

CWD = os.getcwd()

# Loads the CSV file as a numpy matrix
#raw_data = np.genfromtxt(CWD + '/datasets/ionosphere.txt', delimiter=",")

from scipy.io.arff import loadarff
raw_data = loadarff(open(CWD + '/datasets/ionosphere.arff', 'r'))[0]

# Creates the instances and labels sets
instances = list()
labels = list()
for i in range(0, len(raw_data)):
    instance = list()
    for j in range(0, NUM_OF_ATTRIBUTES):
        instance.append(raw_data[i][j])

    instances.append(instance)
    labels.append(int(raw_data[i][NUM_OF_ATTRIBUTES]))