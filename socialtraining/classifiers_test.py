import os
import numpy as np

print('Loading Data Set')

CWD = os.getcwd()

# Loads the CSV file as a numpy matrix
raw_data = np.loadtxt(CWD + '/datasets/diabetes.txt', delimiter=",")

# Creates the instances and labels sets
instances = raw_data[:, 0:8]
labels = raw_data[:, 8]

from sklearn import metrics

#SVM

from sklearn.svm import SVC

print('\nSVM\n')

classifier = SVC()
classifier.fit(instances, labels)

try:
    print(classifier.predict_proba(instances[0]))
except:
    print('SVM Fails')


# LDA

from sklearn.lda import LDA

print('\nLDA\n')

classifier = LDA()
classifier.fit(instances, labels)

print(classifier.predict_proba(instances[0]))

# QDA

from sklearn.qda import QDA

print('\nQDA\n')

classifier = QDA()
classifier.fit(instances, labels)

print(classifier.predict_proba(instances[0]))


# Logistic Regression

from sklearn.linear_model import LogisticRegression

print('\nLogistic Regression\n')

classifier = LogisticRegression()
classifier.fit(instances, labels)

print(classifier.predict_proba(instances[0]))


# KNN

from sklearn.neighbors import KNeighborsClassifier

print('\nK-Nearest Neighbors\n')

classifier = KNeighborsClassifier()
classifier.fit(instances, labels)

print(classifier.predict_proba(instances[0]))


#Decision Trees

from sklearn.tree import DecisionTreeClassifier

print('\nDecision Tree\n')

classifier = DecisionTreeClassifier()
classifier.fit(instances, labels)

print(classifier.predict_proba(instances[0]))


# Kernel Ridge

from sklearn.kernel_ridge import KernelRidge

print('\nKernel Ridge\n')

classifier = KernelRidge()
classifier.fit(instances, labels)

try:
    print(classifier.predict_proba(instances[0]))
except:
    print('Kernel Ridge Fails')

# SGC

from sklearn.linear_model import SGDClassifier

print('\nSGD\n')

classifier = SGDClassifier()
classifier.fit(instances, labels)

try:
    print(classifier.predict_proba(instances[0]))
except:
    print('SGD Fails')



# Gaussian Process

from sklearn.gaussian_process import GaussianProcess

print('\nGaussian Process\n')

classifier = DecisionTreeClassifier()
classifier.fit(instances, labels)

print(classifier.predict_proba(instances[0]))


# Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB

print('\nGaussian Naive Bayes\n')

classifier = GaussianNB()
classifier.fit(instances, labels)

print(classifier.predict_proba(instances[0]))


# Multinomial Naive Bayes

from sklearn.naive_bayes import MultinomialNB

print('\nMultinomial Naive Bayes\n')

classifier = MultinomialNB()
classifier.fit(instances, labels)

print(classifier.predict_proba(instances[0]))


# Bernoulli Naive Bayes

from sklearn.naive_bayes import BernoulliNB

print('\nBernoulli Naive Bayes\n')

classifier = BernoulliNB()
classifier.fit(instances, labels)

print(classifier.predict_proba(instances[0]))


# Random-Forest

from sklearn.ensemble import RandomForestClassifier

print('\nRandom Forest\n')

classifier = RandomForestClassifier()
classifier.fit(instances, labels)

print(classifier.predict_proba(instances[0]))


# Gradient Boosting

from sklearn.ensemble import GradientBoostingClassifier

print('\nGradient Boosting\n')

classifier = GradientBoostingClassifier()
classifier.fit(instances, labels)

print(classifier.predict_proba(instances[0]))