__author__ = 'Matheus Alves'

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from classifier import Classifier
import socialtraining

class ClassifierFactory:
    def create_classifier(classifier_type):

        if classifier_type == \
                socialtraining.ClassifierTypes.linear_discriminant_analysis:
            return Classifier('Linear Discriminant Analysis', LDA())
        elif classifier_type == \
                socialtraining.ClassifierTypes.quadratic_discriminant_analysis:
            return Classifier('Quadratic Discriminant Analysis', QDA())
        elif classifier_type == \
                socialtraining.ClassifierTypes.logistic_regression:
            return Classifier('Logistic Regression', LogisticRegression())
        elif classifier_type == \
                socialtraining.ClassifierTypes.k_nearest_neighbors:
            return Classifier('K-Nearest Neighbors', KNeighborsClassifier())
        elif classifier_type == \
                socialtraining.ClassifierTypes.gaussian_naive_bayes:
            return Classifier('Gaussian Naive Bayes', GaussianNB())
        elif classifier_type == \
                socialtraining.ClassifierTypes.bernoulli_naive_bayes:
            return Classifier('Bernoulli Naive Bayes', BernoulliNB())
        elif classifier_type == \
                socialtraining.ClassifierTypes.gradient_boosting:
            return Classifier('Gradient Boosting',
                               GradientBoostingClassifier())
        elif classifier_type == \
                socialtraining.ClassifierTypes.decision_tree:
            return Classifier('Decision Tree',
                               DecisionTreeClassifier())
