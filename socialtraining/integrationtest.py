import socialtraining
from dataset import *

# Constants
UNLABELED_RATE = UnlabeledDataRates.eighty
# TODO add these as parameter options
from datasetloaders.diabetes import *

# Creates tha data set abstraction class
data_set = DataSet(instances, labels, UNLABELED_RATE)

print('\nPositive class ratio:', data_set.class_ratio[POSITIVE_CLASS_LABEL])
print('Negative class ratio:', data_set.class_ratio[NEGATIVE_CLASS_LABEL])

print('\nTest data size:', len(data_set.test_data.instances))
print('Train data size:', len(data_set.unlabeled_data.instances) +
      len(data_set.labeled_data.instances))

print('\nLabeled data size:', len(data_set.labeled_data.instances))
print('Unlabeled data size:', len(data_set.unlabeled_data.instances))

#Classification
social_training = socialtraining.SocialTraining()

social_training.set_binary_classification_parameters(
    class_threshold=CLASS_THRESHOLD,
    positive_class_label=POSITIVE_CLASS_LABEL,
    negative_class_label=NEGATIVE_CLASS_LABEL)

social_training.set_classifiers([
    socialtraining.ClassifierTypes.linear_discriminant_analysis,
    socialtraining.ClassifierTypes.gradient_boosting,
    socialtraining.ClassifierTypes.logistic_regression,
    socialtraining.ClassifierTypes.k_nearest_neighbors,
    socialtraining.ClassifierTypes.gaussian_naive_bayes,
    socialtraining.ClassifierTypes.quadratic_discriminant_analysis
    ])

social_training.apply_social_training(data_set)
