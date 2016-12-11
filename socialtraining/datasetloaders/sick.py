NUM_OF_ATTRIBUTES = 55

CLASS_THRESHOLD = 0.06

POSITIVE_CLASS_LABEL = 1.
NEGATIVE_CLASS_LABEL = 0.

print('Loading Data Set: ' + "sick")

import os
import numpy as np

HEADERS = list()
HEADERS.append("@relation sick")
HEADERS.append("@attribute 'age' integer")
HEADERS.append("@attribute 'sex' { F, M}")
HEADERS.append("@attribute 'on thyroxine' { f, t}")
HEADERS.append("@attribute 'query on thyroxine' { f, t}")
HEADERS.append("@attribute 'on antithyroid medication' { f, t}")
HEADERS.append("@attribute 'sick' { f, t}")
HEADERS.append("@attribute 'pregnant' { f, t}")
HEADERS.append("@attribute 'thyroid surgery' { f, t}")
HEADERS.append("@attribute 'I131 treatment' { f, t}")
HEADERS.append("@attribute 'query hypothyroid' { f, t}")
HEADERS.append("@attribute 'query hyperthyroid' { f, t}")
HEADERS.append("@attribute 'lithium' { f, t}")
HEADERS.append("@attribute 'goitre' { f, t}")
HEADERS.append("@attribute 'tumor' { f, t}")
HEADERS.append("@attribute 'hypopituitary' { f, t}")
HEADERS.append("@attribute 'psych' { f, t}")
HEADERS.append("@attribute 'TSH measured' { t, f}")
HEADERS.append("@attribute 'TSH' real")
HEADERS.append("@attribute 'T3 measured' { t, f}")
HEADERS.append("@attribute 'T3' real")
HEADERS.append("@attribute 'TT4 measured' { t, f}")
HEADERS.append("@attribute 'TT4' real")
HEADERS.append("@attribute 'T4U measured' { t, f}")
HEADERS.append("@attribute 'T4U' real")
HEADERS.append("@attribute 'FTI measured' { t, f}")
HEADERS.append("@attribute 'FTI' real")
HEADERS.append("@attribute 'TBG measured' { f}")
HEADERS.append("@attribute 'TBG' real")
HEADERS.append("@attribute 'referral source' { SVHC, other, SVI, STMW, SVHD}")
HEADERS.append("@attribute 'Class' { negative, sick}")
HEADERS.append("@data")

CWD = os.getcwd()

from scipy.io.arff import loadarff
raw_data = loadarff(open(CWD + '/datasets/sick.arff', 'r'))[0]

import pandas as pd
df = pd.DataFrame(raw_data)

raw_data = pd.get_dummies(df, dummy_na=True).values

print (raw_data[0])
print (len(raw_data[0]))

# Creates the instances and labels sets
instances = list()
labels = list()
for i in range(0, len(raw_data)):
    instance = list()
    for j in range(0, NUM_OF_ATTRIBUTES):
        instance.append(raw_data[i][j])

    instances.append(instance)
    labels.append(raw_data[i][NUM_OF_ATTRIBUTES])
