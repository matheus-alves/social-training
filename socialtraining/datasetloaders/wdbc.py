NUM_OF_ATTRIBUTES = 30

CLASS_THRESHOLD = 0.37

POSITIVE_CLASS_LABEL = 0
NEGATIVE_CLASS_LABEL = 1

print('Loading Data Set: ' + "wdbc")

import os
import numpy as np

HEADERS = list()
HEADERS.append("@RELATION 'Wisconsin Diagnostic Breast Cancer'")
HEADERS.append("@ATTRIBUTE IDNumber integer")
HEADERS.append("@ATTRIBUTE class {0,1}")
HEADERS.append("@ATTRIBUTE RealValuedInputFeature_1 real")
HEADERS.append("@ATTRIBUTE RealValuedInputFeature_2 real")
HEADERS.append("@ATTRIBUTE RealValuedInputFeature_3 real")
HEADERS.append("@ATTRIBUTE RealValuedInputFeature_4 real")
HEADERS.append("@ATTRIBUTE RealValuedInputFeature_5 real")
HEADERS.append("@ATTRIBUTE RealValuedInputFeature_6 real")
HEADERS.append("@ATTRIBUTE RealValuedInputFeature_7 real")
HEADERS.append("@ATTRIBUTE RealValuedInputFeature_8 real")
HEADERS.append("@ATTRIBUTE RealValuedInputFeature_9 real")
HEADERS.append("@ATTRIBUTE RealValuedInputFeature_10 real")
HEADERS.append("@ATTRIBUTE RealValuedInputFeature_11 real")
HEADERS.append("@ATTRIBUTE RealValuedInputFeature_12 real")
HEADERS.append("@ATTRIBUTE RealValuedInputFeature_13 real")
HEADERS.append("@ATTRIBUTE RealValuedInputFeature_14 real")
HEADERS.append("@ATTRIBUTE RealValuedInputFeature_15 real")
HEADERS.append("@ATTRIBUTE RealValuedInputFeature_16 real")
HEADERS.append("@ATTRIBUTE RealValuedInputFeature_17 real")
HEADERS.append("@ATTRIBUTE RealValuedInputFeature_18 real")
HEADERS.append("@ATTRIBUTE RealValuedInputFeature_19 real")
HEADERS.append("@ATTRIBUTE RealValuedInputFeature_20 real")
HEADERS.append("@ATTRIBUTE RealValuedInputFeature_21 real")
HEADERS.append("@ATTRIBUTE RealValuedInputFeature_22 real")
HEADERS.append("@ATTRIBUTE RealValuedInputFeature_23 real")
HEADERS.append("@ATTRIBUTE RealValuedInputFeature_24 real")
HEADERS.append("@ATTRIBUTE RealValuedInputFeature_25 real")
HEADERS.append("@ATTRIBUTE RealValuedInputFeature_26 real")
HEADERS.append("@ATTRIBUTE RealValuedInputFeature_27 real")
HEADERS.append("@ATTRIBUTE RealValuedInputFeature_28 real")
HEADERS.append("@ATTRIBUTE RealValuedInputFeature_29 real")
HEADERS.append("@ATTRIBUTE RealValuedInputFeature_30 real")
HEADERS.append("@DATA")

CWD = os.getcwd()

# Loads the CSV file as a numpy matrix
#raw_data = np.genfromtxt(CWD + '/datasets/wdbc.txt', delimiter=",")

from scipy.io.arff import loadarff
raw_data = loadarff(open(CWD + '/datasets/wdbc.arff', 'r'))[0]

# Creates the instances and labels sets
instances = list()
labels = list()
for i in range(0, len(raw_data)):
    instance = list()
    for j in range(2, NUM_OF_ATTRIBUTES + 2):
        instance.append(raw_data[i][j])

    instances.append(instance)
    labels.append(int(raw_data[i][1]))