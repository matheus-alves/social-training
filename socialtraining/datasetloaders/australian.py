NUM_OF_ATTRIBUTES = 15

CLASS_THRESHOLD = 0.55

POSITIVE_CLASS_LABEL = b'-'
NEGATIVE_CLASS_LABEL = b'+'

print('Loading Data Set: ' + "australian")

import os
import numpy as np

HEADERS = list()
HEADERS.append("@relation credit-rating")
HEADERS.append("@attribute A1		{b,a}")
HEADERS.append("@attribute A2		REAL")
HEADERS.append("@attribute A3		REAL")
HEADERS.append("@attribute A4		{u, y, l, t}")
HEADERS.append("@attribute A5		{g, p, gg}")
HEADERS.append("@attribute A6		{c, d, cc, i, j, k, m, r, q, w, x, e, aa, ff}")
HEADERS.append("@attribute A7		{v, h, bb, j, n, z, dd, ff, o}")
HEADERS.append("@attribute A8		REAL")
HEADERS.append("@attribute A9		{t,f}")
HEADERS.append("@attribute A10		{t,f}")
HEADERS.append("@attribute A11		REAL")
HEADERS.append("@attribute A12		{t,f}")
HEADERS.append("@attribute A13		{g, p, s}")
HEADERS.append("@attribute A14		REAL")
HEADERS.append("@attribute A15		REAL")
HEADERS.append("@attribute class	{+, -}")
HEADERS.append("@data")

CWD = os.getcwd()

# Loads the CSV file as a numpy matrix
#raw_data = np.loadtxt(CWD + '/datasets/australian.txt', delimiter=" ")

from scipy.io.arff import loadarff
raw_data = loadarff(open(CWD + '/datasets/australian.arff', 'r'))[0]

# Creates the instances and labels sets
instances = list()
labels = list()
for i in range(0, len(raw_data)):
    instance = list()
    for j in range(0, NUM_OF_ATTRIBUTES):
        instance.append(raw_data[i][j])

    instances.append(instance)
    labels.append(raw_data[i][NUM_OF_ATTRIBUTES])

from sklearn.feature_extraction import DictVectorizer
Dvec = DictVectorizer()

Dvec.fit_transform(instances).toarray()

# from sklearn.preprocessing import OneHotEncoder
#
# enc = OneHotEncoder()
# enc.fit(instances)
# OneHotEncoder(categorical_features='all', dtype='string',
#        handle_unknown='error', n_values='auto', sparse=True)
# enc.transform(instances).toarray()

# instances = np.array(instances,
#                      dtype='V, f, f, V, V, V, '
#                            'V, '
#                            'f, V, V, f, V, V, f, f')


# dtype=[('A1', '>b'), ('A2', '>f4'), ('A3', '>f4'),
#                             ('A4', '>b'), ('A5', '>b'), ('A6', '>b'),
#                             ('A7', '>b'), ('A8', '>f4'), ('A9', '>b'),
#                             ('A10', '>b'), ('A11', '>f4'), ('A12', '>b'),
#                             ('A13', '>b'), ('A14', '>f4'), ('A15', '>f4')])