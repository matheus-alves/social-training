NUM_OF_ATTRIBUTES = 74

CLASS_THRESHOLD = 0.52

POSITIVE_CLASS_LABEL = 1.
NEGATIVE_CLASS_LABEL = 0.

print('Loading Data Set: ' + "kr-vs-kp")

import os
import numpy as np

HEADERS = list()
HEADERS.append("@relation kr-vs-kp")
HEADERS.append("@attribute 'bkblk' {'t','f'}")
HEADERS.append("@attribute 'bknwy' {'t','f'}")
HEADERS.append("@attribute 'bkon8' {'t','f'}")
HEADERS.append("@attribute 'bkona' {'t','f'}")
HEADERS.append("@attribute 'bkspr' {'t','f'}")
HEADERS.append("@attribute 'bkxbq' {'t','f'}")
HEADERS.append("@attribute 'bkxcr' {'t','f'}")
HEADERS.append("@attribute 'bkxwp' {'t','f'}")
HEADERS.append("@attribute 'blxwp' {'t','f'}")
HEADERS.append("@attribute 'bxqsq' {'t','f'}")
HEADERS.append("@attribute 'cntxt' {'t','f'}")
HEADERS.append("@attribute 'dsopp' {'t','f'}")
HEADERS.append("@attribute 'dwipd' {'g','l'}")
HEADERS.append("@attribute 'hdchk' {'t','f'}")
HEADERS.append("@attribute 'katri' {'b','n','w'}")
HEADERS.append("@attribute 'mulch' {'t','f'}")
HEADERS.append("@attribute 'qxmsq' {'t','f'}")
HEADERS.append("@attribute 'r2ar8' {'t','f'}")
HEADERS.append("@attribute 'reskd' {'t','f'}")
HEADERS.append("@attribute 'reskr' {'t','f'}")
HEADERS.append("@attribute 'rimmx' {'t','f'}")
HEADERS.append("@attribute 'rkxwp' {'t','f'}")
HEADERS.append("@attribute 'rxmsq' {'t','f'}")
HEADERS.append("@attribute 'simpl' {'t','f'}")
HEADERS.append("@attribute 'skach' {'t','f'}")
HEADERS.append("@attribute 'skewr' {'t','f'}")
HEADERS.append("@attribute 'skrxp' {'t','f'}")
HEADERS.append("@attribute 'spcop' {'t','f'}")
HEADERS.append("@attribute 'stlmt' {'t','f'}")
HEADERS.append("@attribute 'thrsk' {'t','f'}")
HEADERS.append("@attribute 'wkcti' {'t','f'}")
HEADERS.append("@attribute 'wkna8' {'t','f'}")
HEADERS.append("@attribute 'wknck' {'t','f'}")
HEADERS.append("@attribute 'wkovl' {'t','f'}")
HEADERS.append("@attribute 'wkpos' {'t','f'}")
HEADERS.append("@attribute 'wtoeg' {'n','t','f'}")
HEADERS.append("@attribute 'class' {'won','nowin'}")
HEADERS.append("@data")

CWD = os.getcwd()

from scipy.io.arff import loadarff
raw_data = loadarff(open(CWD + '/datasets/kr-vs-kp.arff', 'r'))[0]

import pandas as pd
df = pd.DataFrame(raw_data)

raw_data = pd.get_dummies(df).values

# print (raw_data[0])
# print (len(raw_data[0]))

# Creates the instances and labels sets
instances = list()
labels = list()
for i in range(0, len(raw_data)):
    instance = list()
    for j in range(0, NUM_OF_ATTRIBUTES):
        instance.append(raw_data[i][j])

    instances.append(instance)
    labels.append(raw_data[i][NUM_OF_ATTRIBUTES])
