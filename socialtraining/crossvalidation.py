__author__ = 'Matheus Alves'

"""
This module creates the cross_validation folds.
"""

NUMBER_OF_FOLDS = 10

def generate_fold(instances, labels, fold_number):

    group_size = int(len(instances) / NUMBER_OF_FOLDS)

    start = fold_number * group_size
    end = start + group_size

    fold_instances = list()
    fold_labels = list()

    for i in range (start, end):
        fold_instances.append(instances[i])
        fold_labels.append(labels[i])

    for i in range (end, len(instances)):
        fold_instances.append(instances[i])
        fold_labels.append(labels[i])

    for i in range (0, start):
        fold_instances.append(instances[i])
        fold_labels.append(labels[i])

    fold_data = (fold_instances, fold_labels)

    return fold_data

def generate_cv_pre_post_scf_unlabeled_metrics(metrics):

    print('\nPre-SCF unlabeled metrics:\n')

    cv_pre_average = 0.0
    cv_classifiers_results = dict()
    cv_post_average = 0.0

    for metric in metrics:
        cv_pre_average += metric[0][1]
        cv_post_average += metric[1]
        classifiers_results = metric[0][0]

        for classifier in classifiers_results:
            if classifier in cv_classifiers_results:
                cv_classifiers_results[classifier] += \
                    classifiers_results[classifier]
            else:
                cv_classifiers_results[classifier] = \
                    classifiers_results[classifier]

    for classifier in cv_classifiers_results:
        print(classifier,
              '= {0:0.2f}'.format(cv_classifiers_results[classifier] /
                                  NUMBER_OF_FOLDS))

    print('\nAverage', '= {0:0.2f}'.format(
        (cv_pre_average / NUMBER_OF_FOLDS) / len(cv_classifiers_results)))

    print('\nPost-SCF unlabeled metrics:\n')
    print('Average', '= {0:0.2f}'.format(cv_post_average / NUMBER_OF_FOLDS))

def generate_cv_pre_scf_metrics(metrics):

    print('\nPre-SCF classification:\n')

    cv_pre_average = 0.0
    cv_classifiers_results = dict()

    for metric in metrics:
        cv_pre_average += metric[2][1]
        classifiers_results = metric[2][0]

        for classifier in classifiers_results:
            if classifier in cv_classifiers_results:
                cv_classifiers_results[classifier] += \
                    classifiers_results[classifier]
            else:
                cv_classifiers_results[classifier] = \
                    classifiers_results[classifier]

    for classifier in cv_classifiers_results:
        print(classifier,
              '= {0:0.2f}'.format(cv_classifiers_results[classifier] /
                                  NUMBER_OF_FOLDS))

    print('\nAverage', '= {0:0.2f}'.format(
        (cv_pre_average / NUMBER_OF_FOLDS) / len(cv_classifiers_results)))

def generate_cv_post_scf_metrics(metrics):

    print('\nPost-SCF classification:\n')

    cv_post_average = 0.0
    cv_average_error = 0.0
    cv_classifiers_results = dict()

    for metric in metrics:
        cv_post_average += metric[3][1]
        cv_average_error += metric[3][2]
        classifiers_results = metric[3][0]

        for classifier in classifiers_results:
            if classifier in cv_classifiers_results:
                cv_classifiers_results[classifier] += \
                    classifiers_results[classifier]
            else:
                cv_classifiers_results[classifier] = \
                    classifiers_results[classifier]

    for classifier in cv_classifiers_results:
        print(classifier,
              '= {0:0.2f}'.format(cv_classifiers_results[classifier] /
                                  NUMBER_OF_FOLDS))

    print('\nAverage', '= {0:0.2f}'.format(
        (cv_post_average / NUMBER_OF_FOLDS) / len(cv_classifiers_results)))

    print('\nSocial Training Average Error', '= {0:0.3f}'.format(
        (cv_average_error / NUMBER_OF_FOLDS) / len(cv_classifiers_results)))