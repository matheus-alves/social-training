NUM_OF_ATTRIBUTES = 62

CLASS_THRESHOLD = 0.70

POSITIVE_CLASS_LABEL = 1.
NEGATIVE_CLASS_LABEL = 0.

print('Loading Data Set: ' + "german")

import os
import numpy as np

HEADERS = list()
HEADERS.append("@relation german_credit")
HEADERS.append("@attribute checking_status { '<0', '0<=X<200', '>=200', 'no checking'}")
HEADERS.append("@attribute duration real")
HEADERS.append("@attribute credit_history { 'no credits/all paid', 'all paid', 'existing paid', 'delayed previously', 'critical/other existing credit'}")
HEADERS.append("@attribute purpose { 'new car', 'used car', furniture/equipment, radio/tv, 'domestic appliance', repairs, education, vacation, retraining, business, other}")
HEADERS.append("@attribute credit_amount real")
HEADERS.append("@attribute savings_status { '<100', '100<=X<500', '500<=X<1000', '>=1000', 'no known savings'}")
HEADERS.append("@attribute employment { unemployed, '<1', '1<=X<4', '4<=X<7', '>=7'}")
HEADERS.append("@attribute installment_commitment real")
HEADERS.append("@attribute personal_status { 'male div/sep', 'female div/dep/mar', 'male single', 'male mar/wid', 'female single'}")
HEADERS.append("@attribute other_parties { none, 'co applicant', guarantor}")
HEADERS.append("@attribute residence_since real")
HEADERS.append("@attribute property_magnitude { 'real estate', 'life insurance', car, 'no known property'}")
HEADERS.append("@attribute age real")
HEADERS.append("@attribute other_payment_plans { bank, stores, none}")
HEADERS.append("@attribute housing { rent, own, 'for free'}")
HEADERS.append("@attribute existing_credits real")
HEADERS.append("@attribute job { 'unemp/unskilled non res', 'unskilled resident', skilled, 'high qualif/self emp/mgmt'}")
HEADERS.append("@attribute num_dependents real")
HEADERS.append("@attribute own_telephone { none, yes}")
HEADERS.append("@attribute foreign_worker { yes, no}")
HEADERS.append("@attribute class { good, bad}")
HEADERS.append("@data")

CWD = os.getcwd()

# Loads the CSV file as a numpy matrix
#raw_data = np.genfromtxt(CWD + '/datasets/german.txt', delimiter=" ")

from scipy.io.arff import loadarff
raw_data = loadarff(open(CWD + '/datasets/german.arff', 'r'))[0]

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