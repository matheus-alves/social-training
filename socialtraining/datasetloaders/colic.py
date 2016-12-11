NUM_OF_ATTRIBUTES = 93

CLASS_THRESHOLD = 0.63

POSITIVE_CLASS_LABEL = 0.
NEGATIVE_CLASS_LABEL = 1.

print('Loading Data Set: ' + "colic")

HEADERS = list()
HEADERS.append("@relation horse-colic")
HEADERS.append("@attribute 'surgery' { yes, no}")
HEADERS.append("@attribute 'Age' { adult, young}")
HEADERS.append("@attribute 'rectal_temperature' real")
HEADERS.append("@attribute 'pulse' real")
HEADERS.append("@attribute 'respiratory_rate' real")
HEADERS.append("@attribute 'temp_extremities' { normal, warm, cool, cold}")
HEADERS.append("@attribute 'peripheral_pulse' { normal, increased, reduced, absent}")
HEADERS.append("@attribute 'mucous_membranes' { 'normal pink', 'bright pink', 'pale pink', 'pale cyanotic', 'bright red', 'dark cyanotic'}")
HEADERS.append("@attribute 'capillary_refill_time' { '<3', '>=3', 'meaning unknown'}")
HEADERS.append("@attribute 'pain' { 'alert no pain', depressed, 'intermittent mild pain', 'intermittent severe pain', 'continuous severe pain'}")
HEADERS.append("@attribute 'peristalsis' { hypermotile, normal, hypomotile, absent}")
HEADERS.append("@attribute 'abdominal_distension' { none, slight, moderate, severe}")
HEADERS.append("@attribute 'nasogastric_tube' { none, slight, significant}")
HEADERS.append("@attribute 'nasogastric_reflux' { none, '>1l', '<1l'}")
HEADERS.append("@attribute 'nasogastric_reflux_PH' real")
HEADERS.append("@attribute 'rectal_examination' { normal, increased, decreased, absent}")
HEADERS.append("@attribute 'abdomen' { normal, other, 'firm feces in large intestine', 'distended small intestine', 'distended large intestine'}")
HEADERS.append("@attribute 'packed_cell_volume' real")
HEADERS.append("@attribute 'total_protein' real")
HEADERS.append("@attribute 'abdominocentesis_appearance' { clear, cloudy, serosanguinous}")
HEADERS.append("@attribute 'abdomcentesis_total_protein' real")
HEADERS.append("@attribute 'outcome' { lived, died, euthanized}")
HEADERS.append("@attribute 'surgical_lesion' { yes, no}")
HEADERS.append("@data")

import os
import numpy as np

CWD = os.getcwd()

from scipy.io.arff import loadarff
raw_data = loadarff(open(CWD + '/datasets/colic.arff', 'r'))[0]

import pandas as pd
df = pd.DataFrame(raw_data)

raw_data = pd.get_dummies(df, dummy_na=True).values

print (raw_data)
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

# Loads the CSV file as a numpy matrix
#raw_data = np.genfromtxt(CWD + '/datasets/colic.txt', delimiter=" ")