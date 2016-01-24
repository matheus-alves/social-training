from enum import Enum

__author__ = 'Matheus Alves'

"""
This module contains the entry point of the framework and all of the
configuration related methods.

It contains the SocialTraining class, the SocialChoiceFunctionTypes Enum and
the ClassifierTypes Enum.
"""

class ClassifierTypes(Enum):
    """
    Enum that defines the classification algorithms that the framework offers
    support.
    """
    naive_bayes = 0
    decision_trees = 1

class SocialChoiceFunctionTypes(Enum):
    """
    Enum that defines which social choice functions the framework supports.
    """
    borda = 0
    borda_elim = 1

class SocialTraining:
    """
    This class contains the configuration methods of the framework. It also
    is the entry point of the framework, through the apply_social_training
    method.
    """

    def __init__(self):
        """
        Loads the default configurations for classifiers (all Classifiers) and
        social choice functions (SocialChoiceFunctions.borda).
        """

        self.classifiers = list()
        for classifier in ClassifierTypes:
            self.classifiers.append(classifier)

        self.social_choice_function = SocialChoiceFunctionTypes.borda

    def set_classifiers(self, classifiers):
        """
        This method is used to define which specific classifiers are going
        to be used by the social learning framework. In case it is not
        called, the default configuration is to use all available.

        :param classifiers: A list containing the classifiers that are going
        to be used.
        """

        if len(classifiers) < 2:
            raise ValueError('At least two classifier types should be set.')

        for id in classifiers:
            if id not in ClassifierTypes:
                raise ValueError('Unknown classifier type ' + str(id))

        self.classifiers = classifiers

    def set_social_choice_function(self, social_choice_function):
        """
        This method is used to define the social choice function to be used. In
        case it is not called, the default configuration is to use
        SocialChoiceFunctionTypes.borda.

        :param social_choice_function: The social choice function to be used.
        """

        self.social_choice_function = social_choice_function

    def apply_social_training(data_set):
        """
        Applies the social training for the given data set, using the
        configured machine learning algorithms and social choice function.

        :param data_set: A DataSet type object containing the data set
        information.

        :return: A dictionary containing the instances ids as keys and the
        classification as the value.
        """
        pass
