NUM_OF_ATTRIBUTES = 16

CLASS_THRESHOLD = 0.61

POSITIVE_CLASS_LABEL = b"'democrat'"
NEGATIVE_CLASS_LABEL = b"'republican'"

print('Loading Data Set: ' + "vote")

import os
import numpy as np

HEADERS = list()
HEADERS.append("@relation vote")
HEADERS.append("@attribute 'handicapped-infants' { 'n', 'y'}")
HEADERS.append("@attribute 'water-project-cost-sharing' { 'n', 'y'}")
HEADERS.append("@attribute 'adoption-of-the-budget-resolution' { 'n', 'y'}")
HEADERS.append("@attribute 'physician-fee-freeze' { 'n', 'y'}")
HEADERS.append("@attribute 'el-salvador-aid' { 'n', 'y'}")
HEADERS.append("@attribute 'religious-groups-in-schools' { 'n', 'y'}")
HEADERS.append("@attribute 'anti-satellite-test-ban' { 'n', 'y'}")
HEADERS.append("@attribute 'aid-to-nicaraguan-contras' { 'n', 'y'}")
HEADERS.append("@attribute 'mx-missile' { 'n', 'y'}")
HEADERS.append("@attribute 'immigration' { 'n', 'y'}")
HEADERS.append("@attribute 'synfuels-corporation-cutback' { 'n', 'y'}")
HEADERS.append("@attribute 'education-spending' { 'n', 'y'}")
HEADERS.append("@attribute 'superfund-right-to-sue' { 'n', 'y'}")
HEADERS.append("@attribute 'crime' { 'n', 'y'}")
HEADERS.append("@attribute 'duty-free-exports' { 'n', 'y'}")
HEADERS.append("@attribute 'export-administration-act-south-africa' { 'n', 'y'}")
HEADERS.append("@attribute 'Class' { 'democrat', 'republican'}")
HEADERS.append("@data")

CWD = os.getcwd()

from scipy.io.arff import loadarff
raw_data = loadarff(open(CWD + '/datasets/vote.arff', 'r'))[0]

# Creates the instances and labels sets
instances = list()
labels = list()
for i in range(0, len(raw_data)):
    instance = list()
    for j in range(0, NUM_OF_ATTRIBUTES):
        instance.append(raw_data[i][j])

    instances.append(instance)
    labels.append(raw_data[i][NUM_OF_ATTRIBUTES])