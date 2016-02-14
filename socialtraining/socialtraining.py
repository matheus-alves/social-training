from enum import Enum

from classifierfactory import ClassifierFactory
from socialchoiceengine import SocialChoiceEngine

from sklearn import metrics

__author__ = 'Matheus Alves'

"""
This module contains the entry point of the framework and all of the
configuration related methods.

It contains the SocialTraining class, the SocialChoiceFunctionTypes Enum and
the ClassifierTypes Enum.
"""

_CLASS_THRESHOLD = 0.5

_POSITIVE_CLASS_LABEL = 0
_NEGATIVE_CLASS_LABEL = 1

class ClassifierTypes(Enum):
    """
    Enum that defines the classification algorithms that the framework offers
    support.
    """
    linear_discriminant_analysis = 0
    quadratic_discriminant_analysis = 1
    logistic_regression = 2
    k_nearest_neighbors = 3
    gaussian_naive_bayes = 4
    bernoulli_naive_bayes = 5
    gradient_boosting = 6
    decision_tree = 7

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

        self._classifier_types = list()
        for classifier in ClassifierTypes:
            self._classifier_types.append(classifier)

        self._social_choice_function = SocialChoiceFunctionTypes.borda

        self._positive_class_label = _POSITIVE_CLASS_LABEL
        self._negative_class_label = _NEGATIVE_CLASS_LABEL
        self._class_threshold = _CLASS_THRESHOLD

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

        self._classifier_types = classifiers

    def set_social_choice_function(self, social_choice_function):
        """
        This method is used to define the social choice function to be used. In
        case it is not called, the default configuration is to use
        SocialChoiceFunctionTypes.borda.

        :param social_choice_function: The social choice function to be used.
        """

        self._social_choice_function = social_choice_function

    def set_binary_classification_parameters(self,
                                 positive_class_label=_POSITIVE_CLASS_LABEL,
                                 negative_class_label=_NEGATIVE_CLASS_LABEL,
                                 class_threshold=_CLASS_THRESHOLD):

        self._positive_class_label = positive_class_label
        self._negative_class_label = negative_class_label
        self._class_threshold = class_threshold

    def apply_social_training(self, data_set):
        """
        Applies the social training for the given data set, using the
        configured machine learning algorithms and social choice function.

        :param data_set: A DataSet type object containing the data set
        information.

        :return: A dictionary containing the instances ids as keys and the
        classification as the value.
        """

        self._define_classifiers()
        self._train_classifiers(data_set)

        # Classifies the unlabeled instances according to the dataset properties
        if data_set.number_of_classes > 2:
            return self._classify_multi_class(data_set)
        else:  # binary classification
            return self._classify_binary(data_set)

    def _define_classifiers(self):

        self._classifiers = dict()

        for classifier in self._classifier_types:
            self._classifiers[classifier] = \
                ClassifierFactory.create_classifier(classifier)

    def _train_classifiers(self, data_set):

        for classifier in self._classifier_types:
            self._classifiers[classifier].train(
                data_set.labeled_data.instances, data_set.labeled_data.labels)

    def _classify_multi_class(self, data_set):

        # TODO
        pass

    def _classify_binary(self, data_set):

        pre_scf_unlabeled_metrics = self._generate_pre_scf_unlabeled_metrics(
            data_set)
        scf_engine = SocialChoiceEngine(self._social_choice_function)

        rankings = self._generate_rankings(data_set)
        scf_results = scf_engine.apply_social_choice_function(rankings)
        binary_labels = self._define_binary_labels(scf_results)

        post_scf_unlabeled_metrics = metrics.accuracy_score(
            data_set.unlabeled_data.labels, binary_labels)

        pre_scf_metrics = self._generate_pre_scf_metrics(data_set)

        data_set.labeled_data.instances = data_set.labeled_data.instances + \
                                          data_set.unlabeled_data.instances
        data_set.labeled_data.labels = data_set.labeled_data.labels + \
                                       binary_labels

        self._train_classifiers(data_set)
        post_scf_metrics = self._generate_post_scf_metrics(data_set)

        fold_metrics = (pre_scf_unlabeled_metrics, post_scf_unlabeled_metrics,
                        pre_scf_metrics, post_scf_metrics)

        return fold_metrics

    def _define_binary_labels(self, scf_results):
        positive_threshold = len(scf_results) * self._class_threshold

        scf_labels = dict()
        for position in range(0, len(scf_results)):
            if position < positive_threshold:
                scf_labels[scf_results[position][1]] = \
                    self._positive_class_label
            else:
                scf_labels[scf_results[position][1]] = \
                    self._negative_class_label

        labels = []
        for position in range(0, len(scf_labels)):
            labels.append(scf_labels[str(position)])

        return labels

    def _generate_rankings(self, data_set):

        rankings = dict()

        for classifier_type in self._classifier_types:
            rankings[classifier_type] = \
                self._classifiers[classifier_type].define_ranking(
                    data_set.unlabeled_data.instances)

        return rankings

    def _generate_pre_scf_unlabeled_metrics(self, data_set):

        average = 0.0
        classifiers_results = dict()

        for classifier_type in self._classifier_types:
            classifier = self._classifiers[classifier_type]

            predicted = classifier.classify(data_set.unlabeled_data.instances)
            accuracy = metrics.accuracy_score(data_set.unlabeled_data.labels,
                                              predicted)

            classifiers_results[str(classifier)] = accuracy
            average += accuracy

        pre_scf_unlabeled_metrics = (classifiers_results, average)

        return pre_scf_unlabeled_metrics

    def _generate_pre_scf_metrics(self, data_set):

        average = 0.0
        classifiers_results = dict()

        for classifier_type in self._classifier_types:
            classifier = self._classifiers[classifier_type]

            predicted = classifier.classify(data_set.test_data.instances)
            accuracy = metrics.accuracy_score(data_set.test_data.labels,
                                              predicted)

            classifiers_results[str(classifier)] = accuracy
            average += accuracy

        pre_scf_metrics = (classifiers_results, average)

        return pre_scf_metrics

    def _generate_post_scf_metrics(self, data_set):

        average = 0.0
        average_error = 0.0
        classifiers_results = dict()

        for classifier_type in self._classifier_types:
            classifier = self._classifiers[classifier_type]

            predicted = classifier.classify(data_set.test_data.instances)

            error = 0.0
            for i in range (0, len(data_set.test_data.instances)):
                if predicted[i] != data_set.test_data.labels[i]:
                    error += 1.0

            error /= len(data_set.test_data.instances)
            average_error += error

            accuracy = metrics.accuracy_score(data_set.test_data.labels,
                                              predicted)

            classifiers_results[str(classifier)] = accuracy
            average += accuracy

        post_scf_metrics = (classifiers_results, average, average_error)

        return post_scf_metrics
