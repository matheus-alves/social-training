import time

import socialtraining
import crossvalidation
from dataset import *

# Constants
UNLABELED_RATE = UnlabeledDataRates.eighty
SCF = socialtraining.SocialChoiceFunctionTypes.copeland
# TODO add these as parameter options
from datasetloaders.diabetes import *

start_time = time.clock()

# Creates tha data set abstraction class
data_set = DataSet(instances, labels, UNLABELED_RATE)

print('\nPositive class ratio:', data_set.class_ratio[POSITIVE_CLASS_LABEL])
print('Negative class ratio:', data_set.class_ratio[NEGATIVE_CLASS_LABEL])

print('\nTest data size:', len(data_set.test_data.instances))
print('Train data size:', len(data_set.unlabeled_data.instances) +
      len(data_set.labeled_data.instances))

print('\nUnlabeled data rate:', UNLABELED_RATE)

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
    socialtraining.ClassifierTypes.quadratic_discriminant_analysis,
    socialtraining.ClassifierTypes.decision_tree
    ])

social_training.set_social_choice_function(SCF)

print ('\nApplying Social Training \n')
print ('Social Choice Function: ',
       social_training.get_social_choice_function())

metrics = list()
metrics.append(social_training.apply_social_training(data_set))

for i in range (1, crossvalidation.NUMBER_OF_FOLDS) :
    fold_data = crossvalidation.generate_fold(instances, labels, i)
    data_set = DataSet(fold_data[0], fold_data[1], UNLABELED_RATE)
    metrics.append(social_training.apply_social_training(data_set))

crossvalidation.generate_cv_pre_post_scf_unlabeled_metrics(metrics)
crossvalidation.generate_cv_pre_scf_metrics(metrics)
crossvalidation.generate_cv_post_scf_metrics(metrics)

print ('\nExecution Time: {:.3} seconds'.format(time.clock() - start_time))
