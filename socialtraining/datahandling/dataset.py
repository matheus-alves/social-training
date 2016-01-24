from enum import Enum

__author__ = 'Matheus Alves'

"""
This module contains the DataSet abstraction class. This class was created
to simplify the data set loading process. This module also contains the
UnlabeledDataRates Enum.

TODO license

"""


class UnlabeledDataRates(Enum):
    """
    Enum to define the possible rates for unlabeled data.
    """
    eighty = 0
    sixty = 1
    forty = 2
    twenty = 3

class DataSet:
    """
    This class contains an abstraction of the data set. Its main purpose is
    to define a structure that enables the classifier algorithms to work
    with any external data dynamically.
    """

    def __init__(self, instances, classifications, unlabeled_rate):
        """
        Creates an instance of the DataSet class. Uses the provided
        information to set or to calculate its attributes.

        :param instances: An array, sparse or dense, of size [n_samples,
        n_features] containing all the data set instances.
        :param classifications: An array, containing the labels of the
        instances.
        :param unlabeled_rate: A value from the UnlabeledDataRates Enum that
        defines the ratio of instances from the test set
        """
        # TODO finish implementation and documentation
        pass
