__author__ = 'Matheus Alves'

"""
This module creates the cross_validation folds.
"""

import numpy

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

    std_dev_pre_average = list()
    std_dev_classifiers_results = dict()
    std_dev_post_average = list()

    for metric in metrics:
        cv_pre_average += metric[0][1]
        std_dev_pre_average.append(metric[0][1])

        cv_post_average += metric[1]
        std_dev_post_average.append(metric[1])

        classifiers_results = metric[0][0]

        for classifier in classifiers_results:
            if classifier in cv_classifiers_results:
                cv_classifiers_results[classifier] += \
                    classifiers_results[classifier]
            else:
                cv_classifiers_results[classifier] = \
                    classifiers_results[classifier]

                std_dev_classifiers_results[classifier] = list()

            std_dev_classifiers_results[classifier]\
                .append(classifiers_results[classifier])

    for classifier in cv_classifiers_results:
        print(classifier,
              '= {0:0.2f}'.format(cv_classifiers_results[classifier] /
                                  NUMBER_OF_FOLDS), '({0:0.2f})'.format(
                numpy.std(std_dev_classifiers_results[classifier],
                          dtype=numpy.float64)))

    print('\nAverage', '= {0:0.2f}'.format(
        (cv_pre_average / NUMBER_OF_FOLDS) / len(cv_classifiers_results)),
          '({0:0.2f})'.format(numpy.std(std_dev_pre_average,
                                        dtype=numpy.float64)
                              / len(cv_classifiers_results)))

    print('\nPost-SCF unlabeled metrics:\n')
    print('Average', '= {0:0.2f}'.format(cv_post_average / NUMBER_OF_FOLDS),
          '({0:0.2f})'.format(numpy.std(std_dev_post_average,
                                        dtype=numpy.float64)))

def generate_cv_pre_scf_metrics(metrics):

    print('\nPre-SCF classification:\n')

    cv_pre_average = 0.0
    cv_classifiers_results = dict()

    std_dev_pre_average = list()
    std_dev_classifiers_results = dict()

    for metric in metrics:
        cv_pre_average += metric[2][1]
        std_dev_pre_average.append(metric[2][1])

        classifiers_results = metric[2][0]

        for classifier in classifiers_results:
            if classifier in cv_classifiers_results:
                cv_classifiers_results[classifier] += \
                    classifiers_results[classifier]
            else:
                cv_classifiers_results[classifier] = \
                    classifiers_results[classifier]

                std_dev_classifiers_results[classifier] = list()

            std_dev_classifiers_results[classifier]\
                .append(classifiers_results[classifier])

    for classifier in cv_classifiers_results:
        print(classifier,
              '= {0:0.2f}'.format(cv_classifiers_results[classifier] /
                                  NUMBER_OF_FOLDS), '({0:0.2f})'.format(
                numpy.std(std_dev_classifiers_results[classifier],
                          dtype=numpy.float64)))

    print('\nAverage', '= {0:0.2f}'.format(
        (cv_pre_average / NUMBER_OF_FOLDS) / len(cv_classifiers_results)),
          '({0:0.2f})'.format(numpy.std(std_dev_pre_average,
                                        dtype=numpy.float64)
                              / len(cv_classifiers_results)))

def generate_cv_post_scf_metrics(metrics):

    print('\nPost-SCF classification:\n')

    cv_post_average = 0.0
    cv_average_error = 0.0
    cv_classifiers_results = dict()

    std_dev_post_average = list()
    std_dev_average_error = list()
    std_dev_classifiers_results = dict()

    for metric in metrics:
        cv_post_average += metric[3][1]
        std_dev_post_average.append(metric[3][1])

        cv_average_error += metric[3][2]
        std_dev_average_error.append(metric[3][2])

        classifiers_results = metric[3][0]

        for classifier in classifiers_results:
            if classifier in cv_classifiers_results:
                cv_classifiers_results[classifier]['accuracy'] += \
                    classifiers_results[classifier]['accuracy']
                cv_classifiers_results[classifier]['f-score'] += \
                    classifiers_results[classifier]['f-score']
                cv_classifiers_results[classifier]['precision'] += \
                    classifiers_results[classifier]['precision']
                cv_classifiers_results[classifier]['recall'] += \
                    classifiers_results[classifier]['recall']
            else:
                cv_classifiers_results[classifier] = \
                    classifiers_results[classifier]

                std_dev_classifiers_results[classifier] = dict()
                std_dev_classifiers_results[classifier]['accuracy'] = list()
                std_dev_classifiers_results[classifier]['f-score'] = list()
                std_dev_classifiers_results[classifier]['precision'] = list()
                std_dev_classifiers_results[classifier]['recall'] = list()

            std_dev_classifiers_results[classifier]['accuracy']\
                .append(classifiers_results[classifier]['accuracy'])
            std_dev_classifiers_results[classifier]['f-score']\
                .append(classifiers_results[classifier]['f-score'])
            std_dev_classifiers_results[classifier]['precision']\
                .append(classifiers_results[classifier]['precision'])
            std_dev_classifiers_results[classifier]['recall']\
                .append(classifiers_results[classifier]['recall'])

    for classifier in cv_classifiers_results:
        print(classifier,
              ':\n\t Accuracy : {0:0.2f}'.format(cv_classifiers_results[
                                                  classifier]
                                  ['accuracy'] /
                                  NUMBER_OF_FOLDS), '({0:0.2f})'.format(
                numpy.std(std_dev_classifiers_results[classifier]['accuracy'],
                          dtype=numpy.float64)),
              ' | F-Score : {0:0.2f}'.format(cv_classifiers_results[
                                                  classifier]
                                  ['f-score'] /
                                  NUMBER_OF_FOLDS), '({0:0.2f})'.format(
                numpy.std(std_dev_classifiers_results[classifier]['f-score'],
                          dtype=numpy.float64)),
              ' | Precision : {0:0.2f}'.format(cv_classifiers_results[
                                                  classifier]
                                  ['precision'] /
                                  NUMBER_OF_FOLDS), '({0:0.2f})'.format(
                numpy.std(std_dev_classifiers_results[classifier]['precision'],
                          dtype=numpy.float64)),
              ' | Recall : {0:0.2f}'.format(cv_classifiers_results[
                                                  classifier]
                                  ['recall'] /
                                  NUMBER_OF_FOLDS), '({0:0.2f})'.format(
                numpy.std(std_dev_classifiers_results[classifier]['recall'],
                          dtype=numpy.float64)))

    print('\nAverage Accuracy', '= {0:0.2f}'.format(
        (cv_post_average / NUMBER_OF_FOLDS) / len(cv_classifiers_results)),
          '({0:0.2f})'.format(numpy.std(std_dev_post_average,
                                        dtype=numpy.float64)
                              / len(cv_classifiers_results)))

    print('\nSocial Training Average Error', '= {0:0.3f}'.format(
        (cv_average_error / NUMBER_OF_FOLDS) / len(cv_classifiers_results)),
          '({0:0.3f})'.format(numpy.std(std_dev_average_error,
                                        dtype=numpy.float64)
                              / len(cv_classifiers_results)))
