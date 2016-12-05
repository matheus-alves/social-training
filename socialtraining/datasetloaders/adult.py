NUM_OF_ATTRIBUTES = 14

CLASS_THRESHOLD = 0.9995

POSITIVE_CLASS_LABEL = 0
NEGATIVE_CLASS_LABEL = 1

print('Loading Data Set: ' + "adult")

import os
import numpy as np

HEADERS = list()
HEADERS.append("@relation census")
HEADERS.append("@attribute 'age' real")
HEADERS.append("@attribute 'plas' real")
HEADERS.append("@attribute 'pres' real")
HEADERS.append("@attribute 'skin' real")
HEADERS.append("@attribute 'insu' real")
HEADERS.append("@attribute 'mass' real")
HEADERS.append("@attribute 'pedi' real")
HEADERS.append("@attribute 'age' real")
HEADERS.append("@attribute 'class' {0, 1}")
HEADERS.append("@data")

CWD = os.getcwd()

# Loads the CSV file as a numpy matrix
# raw_data = np.loadtxt(CWD + '/datasets/adult.txt', delimiter=",")
import pandas as pd
df = pd.read_csv(open(CWD + '/datasets/adult.txt', 'r'),
                 header=None)

raw_data = pd.get_dummies(df).values

# from scipy.io.arff import loadarff
# raw_data = loadarff(open(CWD + '/datasets/diabetes.arff', 'r'))[0]

# Creates the instances and labels sets
instances = list()
labels = list()
for i in range(0, len(raw_data)):
    instance = list()
    for j in range(0, NUM_OF_ATTRIBUTES):
        instance.append(raw_data[i][j])

    instances.append(instance)
    labels.append(int(raw_data[i][NUM_OF_ATTRIBUTES]))