NUM_OF_ATTRIBUTES = 9

CLASS_THRESHOLD = 0.65

POSITIVE_CLASS_LABEL = b'positive'
NEGATIVE_CLASS_LABEL = b'negative'

print('Loading Data Set: ' + "tic-tac-toe")

import os
import numpy as np

HEADERS = list()
HEADERS.append("@relation tic-tac-toe")
HEADERS.append("@attribute top-left-square {b,o,x}")
HEADERS.append("@attribute top-middle-square {b,o,x}")
HEADERS.append("@attribute top-right-square {b,o,x}")
HEADERS.append("@attribute middle-left-square {b,o,x}")
HEADERS.append("@attribute middle-middle-square {b,o,x}")
HEADERS.append("@attribute middle-right-square {b,o,x}")
HEADERS.append("@attribute bottom-left-square {b,o,x}")
HEADERS.append("@attribute bottom-middle-square {b,o,x}")
HEADERS.append("@attribute bottom-right-square {b,o,x}")
HEADERS.append("@attribute Class {negative,positive}")
HEADERS.append("@data")

CWD = os.getcwd()

from scipy.io.arff import loadarff
raw_data = loadarff(open(CWD + '/datasets/tic-tac-toe.arff', 'r'))[0]

# Creates the instances and labels sets
instances = list()
labels = list()
for i in range(0, len(raw_data)):
    instance = list()
    for j in range(0, NUM_OF_ATTRIBUTES):
        instance.append(raw_data[i][j])

    instances.append(instance)
    labels.append(raw_data[i][NUM_OF_ATTRIBUTES])