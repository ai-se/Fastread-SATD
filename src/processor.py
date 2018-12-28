import random
import time
from copy import copy

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeRegressor

from act_learn import fastread
from tuner import TUNER
import numpy as np
import pandas

import scipy.sparse as sp


import logging
logger = logging.getLogger(__name__)

BEST_CONF_TEMP = [47.96, 'poly', 0.84, 0.75]


def k_fold_with_tuning(test_data, train_data, fold, pool_size, init_pool, budget, label=''):
    """
    We split the train_data into train set and tune set and do our hyperparameter optimization on train and tune. Then,
    for each fold, we predict the final F score for our Test data.
    Stratification attempts to maintain the realtive ratios of positive and negative classes in each of folds.
    This is very helpful for skewed datasets.
    :param test_data: DATA this will not be used on Tuning SVM Hyperparameter. Will only be used at the end for our F score
    :param train_data: DATA this dataset will be split into train and tune set to do hyperparameter tuning.
    :param fold: int number of folds
    :return:
    """
    logger.info(label + " | Starting our " + str(fold) + " fold cross validation for " + test_data.data_pd.iloc[0]['projectname'])

    skfolds = StratifiedKFold(n_splits=fold, random_state=0)

    y_train = train_data.data_pd.loc[:, 'label']
    y_test = test_data.data_pd.loc[:, 'label']

    # This is a list of confusion matrix recieved from each fold.
    conf_mat = []
    clfs = []

    fold_num = 0

    for train_index, tune_index in skfolds.split(train_data.csr_mat, y_train):
        logger.info(label + " | Starting a fold")
        # Training data
        # -------------
        x_train_folds = train_data.csr_mat[train_index]
        y_train_folds = y_train.iloc[train_index]
         # Tuning data
        # ------------
        x_tune_folds = train_data.csr_mat[tune_index]
        y_tune_folds = y_train.iloc[tune_index]

        # Tune with FLASH over train and tune dataset and return the optimized clf that is already fitted
        # -------------------------------------
        # TODO TESTING ACTIVE LEARNING, So, using a custom HARD CODED best_config
        # best_config = tune_with_flash(x_train_folds, y_train_folds, x_tune_folds, y_tune_folds, fold_num,
        #                     pool_size, init_pool, budget, label)

        # Prior to active learning WITH PROBABILITY TURNED TRUE FOR ACTIVE LEARNING
        clf = SVC(C=BEST_CONF_TEMP[0], kernel=BEST_CONF_TEMP[1], gamma=BEST_CONF_TEMP[2], coef0=BEST_CONF_TEMP[3],
                  probability=True, random_state=0)


        # Testing active learning now
        fastread(clf, test_data, x_train_folds, y_train_folds)

        # # run optimized clf on the test data and record the confusion matrix
        # -------------------------------------
        y_test_pred = clf.predict(test_data.csr_mat)
        mat = confusion_matrix(y_test, y_test_pred)
        logger.info(label + " | Before Active Learning Optimized F Score for fold " + str(fold_num) + ": " + str(calc_f(mat)))

        # active learning
        # mat = active_learning(best_config, x_train_folds, y_train_folds, test_data.csr_mat, y_test)


        conf_mat.append(mat)
        #clfs.append(clf)
        logger.info(label + " | After Active Learning Optimized F Score for fold " + str(fold_num) + ": " + str(calc_f(mat)))

        fold_num += 1

    return conf_mat, clfs


def tune_with_flash(x_train, y_train, x_tune, y_tune, fold_num, pool_size, init_pool, budget, label=''):
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    tuner = TUNER(fold_num)
    random.seed(fold_num)
    this_budget = budget

    # Make initial population
    param_search_space = tuner.generate_param_pools(pool_size)

    # Evaluate initial pool
    evaluted_configs = random.sample(param_search_space, init_pool)
    param_search_space = list(set(param_search_space).difference(set(evaluted_configs)))

    f_scores = [measure_fitness(x_train, y_train, x_tune, y_tune, configs) for configs in evaluted_configs]

    # Filtering NaN case
    evaluted_configs, f_scores = filter_no_info(label, evaluted_configs, f_scores)


    logger.info(label + " | F Score of init pool: " + str(f_scores))

    # hold best values
    ids = np.argsort(f_scores)[::-1][:1]
    best_f = f_scores[ids[0]]
    best_config = evaluted_configs[ids[0]]

    # converting str value to int for CART to work
    evaluted_configs = [(x[0], tuner.label_transform(x[1]), x[2], x[3]) for x in evaluted_configs]
    param_search_space = [(x[0], tuner.label_transform(x[1]), x[2], x[3]) for x in param_search_space]

    # number of eval
    eval = 0
    while this_budget > 0:
        cart_model = DecisionTreeRegressor(random_state=1)

        cart_model.fit(evaluted_configs, f_scores)

        next_config_id = acquisition_fn(param_search_space, cart_model)
        next_config = param_search_space.pop(next_config_id)

        next_config_normal = (next_config[0], tuner.label_reverse_transform(next_config[1]), next_config[2], next_config[3])

        next_f = measure_fitness(x_train, y_train, x_tune, y_tune, next_config_normal)

        if np.isnan(next_f) or next_f == 0:
            continue

        f_scores.append(next_f)
        evaluted_configs.append(next_config)

        if isBetter(next_f, best_f):
            best_config = next_config_normal
            best_f = next_f
            this_budget += 1
            logger.info(label + " | new F: " + str(best_f) + " budget " + str(this_budget))
        this_budget -= 1
        eval += 1

    logger.info(label+ " | Eval: " + str(eval))


    # Going to return Config so that I can use SVR on the outside
    # clf = SVC(C=best_config[0], kernel=best_config[1], gamma=best_config[2], coef0=best_config[3], random_state=0)
    # clf.fit(x_train, y_train)

    return best_config


def acquisition_fn(search_space, cart_model):
    predicted = cart_model.predict(search_space)

    ids = np.argsort(predicted)[::-1][:1]
    val = predicted[ids[0]]
    return ids[0]
    # val = 0
    # id = 0
    # for i, x in enumerate(predicted):
    #     if val < x:
    #         val = x
    #         id = i
    # return id

def isBetter(new, old):
    return old < new

def measure_fitness(x_train, y_train, x_tune, y_tune, configs):
    clf = SVC(C=configs[0], kernel=configs[1], gamma=configs[2], coef0=configs[3], random_state=0)

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_tune)
    cmat = confusion_matrix(y_tune, y_pred)

    return calc_f(cmat)


def calc_f(cmat):
    # Precision
    # ---------
    prec = cmat[1, 1] / (cmat[1, 1] + cmat[0, 1])

    # Recall
    # ------
    recall = cmat[1, 1] / (cmat[1, 1] + cmat[1, 0])

    # F1 Score
    # --------
    f1 = 2 * (prec * recall) / (prec + recall)

    return f1


def filter_no_info(label, evaluated_configs, fscores):
    for i, score in enumerate(fscores):
        if np.isnan(score) or score == 0:
            del evaluated_configs[i]
            del fscores[i]
            logger.info(label + "| filtered one: " + str(fscores))

    return evaluated_configs, fscores


def active_learning(best_config, x_train_folds, y_train_folds, x_test, y_test):
    y_train_folds = y_train_folds.replace('yes', 1.0)
    y_train_folds = y_train_folds.replace('no', -1.0)

    clf = SVR(C=best_config[0], kernel=best_config[1], gamma=best_config[2], coef0=best_config[3])
    clf.fit(x_train_folds, y_train_folds)
    # # run optimized clf on the test data and record the confusion matrix
    # -------------------------------------
    y_test_pred = clf.predict(x_test)

    y_test_pred2 = np.where(y_test_pred > .5, 'yes', (np.where(y_test_pred > -.5, 'undefined', 'no')))

    ids = np.argwhere(y_test_pred2=='undefined')
    ids = ids.reshape(-1)
    rand_ids = np.random.choice(ids, 20)

    x_train_folds = sp.vstack((x_train_folds, x_test[rand_ids]))
    y_train_folds = pandas.concat((y_train_folds, y_test[rand_ids]))





    y_train_folds = y_train_folds.replace('yes', 1.0)
    y_train_folds = y_train_folds.replace('no', -1.0)

    clf = SVR(C=best_config[0], kernel=best_config[1], gamma=best_config[2], coef0=best_config[3])
    clf.fit(x_train_folds, y_train_folds)
    # # run optimized clf on the test data and record the confusion matrix
    # -------------------------------------
    y_test_pred = clf.predict(x_test)

    y_test_pred2 = np.where(y_test_pred > .5, 'yes', (np.where(y_test_pred > -.5, 'undefined', 'no')))

    ids = np.argwhere(y_test_pred2 == 'undefined')
    ids = ids.reshape(-1)
    rand_ids = np.random.choice(ids, 20)

    x_train_folds = sp.vstack((x_train_folds, x_test[rand_ids]))
    y_train_folds = pandas.concat((y_train_folds, y_test[rand_ids]))




    y_train_folds = y_train_folds.replace('yes', 1.0)
    y_train_folds = y_train_folds.replace('no', -1.0)

    clf = SVR(C=best_config[0], kernel=best_config[1], gamma=best_config[2], coef0=best_config[3])
    clf.fit(x_train_folds, y_train_folds)
    # # run optimized clf on the test data and record the confusion matrix
    # -------------------------------------
    y_test_pred = clf.predict(x_test)

    y_test_pred2 = np.where(y_test_pred > .5, 'yes', (np.where(y_test_pred > -.5, 'undefined', 'no')))

    ids = np.argwhere(y_test_pred2 == 'undefined')
    ids = ids.reshape(-1)
    rand_ids = np.random.choice(ids, 20)

    x_train_folds = sp.vstack((x_train_folds, x_test[rand_ids]))
    y_train_folds = pandas.concat((y_train_folds, y_test[rand_ids]))




    y_train_folds = y_train_folds.replace(1.0, 'yes')
    y_train_folds = y_train_folds.replace(-1.0, 'no')

    clf = SVC(C=best_config[0], kernel=best_config[1], gamma=best_config[2], coef0=best_config[3], random_state=0)
    clf.fit(x_train_folds, y_train_folds)
    y_test_pred = clf.predict(x_test)
    mat = confusion_matrix(y_test, y_test_pred)
    print(calc_f(mat))
    return mat