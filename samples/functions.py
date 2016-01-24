from enum import Enum


# TODO add these functions to the proper file in te socialtraining project

# DATA_SET_HANDLING


_TEST_GROUP_RATE = 0.25


class UnlabeledDataRates(Enum):
    eighty = 0
    sixty = 1
    forty = 2
    twenty = 3


class DataSet:
    class _DataGroup:

        def __init__(self, instances, labels):
            self.instances = instances
            self.labels = labels

    def __init__(self, instances, labels,
                 unlabeled_data_rate=UnlabeledDataRates.twenty,
                 test_group_rate=_TEST_GROUP_RATE):

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

# CLASSIFIERS

from sklearn import metrics


class _Classifier:
    def __init__(self, id, algorithm):
        self._id = id
        self._algorithm = algorithm

    def __str__(self):
        return self._id

    def train(self, instances, labels):
        self._algorithm.fit(instances, labels)

    def classify(self, instances):
        return self._algorithm.predict(instances)

    def define_ranking(self, instances):
        ranking = []
        for position in range(0, len(instances)):
            prob = self._algorithm.predict_proba(instances[position])
            ranking.append((position, prob[0][0]))

        ranking.sort(key=lambda tup: tup[1], reverse=True)

        return ranking

    def define_class_ranking(self, instances):
        # TODO
        pass


from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier


class _ClassifierFactory:
    def create_classifier(classifier_type):

        if classifier_type == ClassifierTypes.linear_discriminant_analysis:
            return _Classifier('Linear Discriminant Analysis', LDA())
        elif classifier_type == \
                ClassifierTypes.quadratic_discriminant_analysis:
            return _Classifier('Quadratic Discriminant Analysis', QDA())
        elif classifier_type == ClassifierTypes.logistic_regression:
            return _Classifier('Logistic Regression', LogisticRegression())
        elif classifier_type == ClassifierTypes.k_nearest_neighbors:
            return _Classifier('K-Nearest Neighbors', KNeighborsClassifier())
        elif classifier_type == ClassifierTypes.gaussian_naive_bayes:
            return _Classifier('Gaussian Naive Bayes', GaussianNB())
        elif classifier_type == ClassifierTypes.bernoulli_naive_bayes:
            return _Classifier('Bernoulli Naive Bayes', BernoulliNB())
        elif classifier_type == ClassifierTypes.gradient_boosting:
            return _Classifier('Gradient Boosting',
                               GradientBoostingClassifier())

# SOCIAL CHOICE

from ballotbox.ballot import BallotBox

from ballotbox.singlewinner.preferential.borda import BordaVoting


class _SocialChoiceFactory:
    def create_engine(social_choice_function):
        if social_choice_function == SocialChoiceFunctionTypes.borda:
            return (BallotBox(method=BordaVoting, mode="standard"),
                    'Borda Count')


class _SocialChoiceEngine:
    def __init__(self, social_choice_function):

        factory_result = _SocialChoiceFactory.create_engine(
            social_choice_function)

        self._engine = factory_result[0]
        self._method = factory_result[1]

    def apply_social_choice_function(self, rankings):

        print('\nApplying Social Choice Function: ', self._method)

        self._extract_votes_from_rankings(rankings)

        return self._engine.get_winner()[0]

    def _extract_votes_from_rankings(self, rankings):

        for classifier in rankings:
            ranking = rankings[classifier]
            preferences = dict()

            preference = 1
            for position in range(0, len(ranking)):
                preferences[ranking[position][0]] = preference
                preference += 1

            self._engine.add_vote(preferences)

# SOCIAL_TRAINING

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
                _ClassifierFactory.create_classifier(classifier)

    def _train_classifiers(self, data_set):

        for classifier in self._classifier_types:
            self._classifiers[classifier].train(
                data_set.labeled_data.instances, data_set.labeled_data.labels)

    def _classify_multi_class(self, data_set):

        # TODO
        pass

    def _classify_binary(self, data_set):

        self._generate_pre_scf_unlabeled_metrics(data_set)
        scf_engine = _SocialChoiceEngine(self._social_choice_function)

        rankings = self._generate_rankings(data_set)
        scf_results = scf_engine.apply_social_choice_function(rankings)
        binary_labels = self._define_binary_labels(scf_results)

        print('\nPost-SCF unlabeled metrics:\n')
        print('{0:0.2f}'.format(metrics.accuracy_score(
            data_set.unlabeled_data.labels, binary_labels)))

        self._generate_pre_scf_metrics(data_set)

        data_set.labeled_data.instances = data_set.labeled_data.instances + \
                                          data_set.unlabeled_data.instances
        data_set.labeled_data.labels = data_set.labeled_data.labels + \
                                       binary_labels

        self._train_classifiers(data_set)
        self._generate_post_scf_metrics(data_set)

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

        print('\nPre-SCF unlabeled metrics:\n')

        average = 0.0

        for classifier_type in self._classifier_types:
            classifier = self._classifiers[classifier_type]

            predicted = classifier.classify(data_set.unlabeled_data.instances)
            accuracy = metrics.accuracy_score(data_set.unlabeled_data.labels,
                                              predicted)

            print(str(classifier), '= {0:0.2f}'.format(accuracy))
            average += accuracy

        print('\nAverage', '= {0:0.2f}'.format(
            average / len(self._classifier_types)))

    def _generate_pre_scf_metrics(self, data_set):

        print('\nPre-SCF classification:\n')

        average = 0.0

        for classifier_type in self._classifier_types:
            classifier = self._classifiers[classifier_type]

            predicted = classifier.classify(data_set.test_data.instances)
            accuracy = metrics.accuracy_score(data_set.test_data.labels,
                                              predicted)

            print(str(classifier), '= {0:0.2f}'.format(accuracy))
            average += accuracy

        print('\nAverage', '= {0:0.2f}'.format(
            average / len(self._classifier_types)))

    def _generate_post_scf_metrics(self, data_set):

        print('\nPost-SCF classification:\n')

        average = 0.0

        for classifier_type in self._classifier_types:
            classifier = self._classifiers[classifier_type]

            predicted = classifier.classify(data_set.test_data.instances)
            accuracy = metrics.accuracy_score(data_set.test_data.labels,
                                              predicted)

            print(str(classifier), '= {0:0.2f}'.format(accuracy))
            average += accuracy

        print('\nAverage', '= {0:0.2f}'.format(
            average / len(self._classifier_types)))
