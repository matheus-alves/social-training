from enum import Enum

__author__ = 'Matheus Alves'

"""
This module contains the DataSet abstraction class. This class was created
to simplify the data set loading process. This module also contains the
UnlabeledDataRates Enum.
"""

_TEST_GROUP_RATE = 0.25

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

    class _DataGroup:

        def __init__(self, instances, labels):
            self.instances = instances
            self.labels = labels

    def __init__(self, instances, labels,
                 unlabeled_data_rate=UnlabeledDataRates.twenty,
                 test_group_rate=_TEST_GROUP_RATE):
        """
        Creates an instance of the DataSet class. Uses the provided
        information to set or to calculate its attributes.

        :param instances: An array, sparse or dense, of size [n_samples,
        n_features] containing all the data set instances.
        :param classifications: An array, containing the labels of the
        instances.
        :param unlabeled_rate: A value from the UnlabeledDataRates Enum that
        defines the ratio of instances from the test set
        :param test_group_rate: A float value that determines the percentage of
        data that is going to be used for the test group. It has a default
        of 25%
        """

        self._extract_class_info(labels)
        self._generate_groups(instances, labels,
                              self._extract_unlabeled_data_rate(
                                  unlabeled_data_rate),
                              test_group_rate)

    def _extract_class_info(self, labels):

        # Calculates the positive and negative groups sizes
        # in order to keep the same ratio
        self.class_count = dict()

        for label in labels:
            if label in self.class_count:
                self.class_count[label] += 1
            else:
                self.class_count[label] = 1

        self.number_of_classes = len(self.class_count)

        self.class_ratio = dict()

        for label in self.class_count:
            self.class_ratio[label] = self.class_count[label] / len(labels)

    def _extract_unlabeled_data_rate(self, unlabeled_data_rate):
        if unlabeled_data_rate == UnlabeledDataRates.eighty:
            return 0.8
        elif unlabeled_data_rate == UnlabeledDataRates.sixty:
            return 0.6
        elif unlabeled_data_rate == UnlabeledDataRates.forty:
            return 0.4
        elif unlabeled_data_rate == UnlabeledDataRates.twenty:
            return 0.2

    def _generate_groups(self, instances, labels, unlabeled_data_rate,
                         test_group_rate):

        # Creates the train and test groups keeping the pos/neg ratio
        test_train = self._generate_group_split(test_group_rate, instances,
                                                labels)
        self.test_data = test_train[0]
        train_data = test_train[1]

        unlabeled_labeled = self._generate_group_split(unlabeled_data_rate,
                                                       train_data.instances,
                                                       train_data.labels)
        self.unlabeled_data = unlabeled_labeled[0]
        self.labeled_data = unlabeled_labeled[1]

    def _generate_group_split(self, split_rate, instances, labels):

        split_size = int(len(instances) * split_rate)

        group_a_instances = []
        group_a_labels = []
        group_b_instances = []
        group_b_labels = []

        class_ratios_max = dict()
        for label in self.class_ratio:
            class_ratios_max[label] = \
                int(split_size * self.class_ratio[label])

        class_ratios_count = dict()
        for label in self.class_ratio:
            class_ratios_count[label] = 0

        position = 0
        while position < len(instances):
            label = labels[position]

            if class_ratios_count[label] < class_ratios_max[label]:
                group_a_instances.append(instances[position])
                group_a_labels.append(labels[position])
                class_ratios_count[label] += 1
            else:
                group_b_instances.append(instances[position])
                group_b_labels.append(labels[position])

            position += 1

        data_group_a = self._DataGroup(group_a_instances, group_a_labels)
        data_group_b = self._DataGroup(group_b_instances, group_b_labels)

        groups = (data_group_a, data_group_b)

        return groups
