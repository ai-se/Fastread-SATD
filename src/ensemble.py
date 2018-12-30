import pandas

import numpy as np
from pandas.core.common import SettingWithCopyWarning
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import logging
import scipy.sparse as sp

from processor import calc_f, tune_with_flash

logger = logging.getLogger(__name__)

BEST_CONF_TEMP = [12.23, 'rbf', 0.82, 0.28]
DELTA = .01
PROBA_DELTA = .1
def ensemble_vote(training_data, test_data, project_name, all_datasets):
    import warnings
    warnings.filterwarnings("ignore", category=SettingWithCopyWarning)


    y_test = test_data.data_pd.loc[:, 'label']
    test_data.data_pd.loc[:, 'yes_vote'] = 0
    test_data.data_pd.loc[:, 'no_vote'] = 0

    test_data.data_pd.loc[:, 'total_dataset'] = ''
    test_data.data_pd.loc[:, 'distance_from_plane'] = 0
    test_data.data_pd.loc[:, 'code_ensemble'] = ''
    test_data.data_pd.loc[:, 'probability'] = 0
    test_data.data_pd.loc[:, 'code_combined'] = ''


    for train_dataset_name in all_datasets:
        if train_dataset_name in project_name:
            continue
        training_ids = training_data.data_pd.loc[training_data.data_pd['projectname'] == train_dataset_name]
        training_ids = list(training_ids.index)

        x_train = training_data.csr_mat[training_ids]
        y_train = training_data.data_pd.loc[training_ids, 'label']

        sss = StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=0)
        for train_index, tune_index in sss.split(x_train, y_train):
            x_train_flash, x_tune_flash = x_train[train_index], x_train[tune_index]
            y_train_flash, y_tune_flash = y_train.iloc[train_index], y_train.iloc[tune_index]
            best_conf = tune_with_flash(x_train_flash, y_train_flash, x_tune_flash, y_tune_flash, fold_num=1, pool_size=100000,
                            init_pool=10, budget=10, label=project_name)

        clf = SVC(C=best_conf[0], kernel=best_conf[1], gamma=best_conf[2], coef0=best_conf[3],
                  probability=False, random_state=0)
        clf.fit(x_train, y_train)

        y_pred = clf.predict(test_data.csr_mat)

        test_data.data_pd.loc[:, 'current_pred'] = y_pred.tolist()

        yes_ids = test_data.data_pd[test_data.data_pd.loc[:, "current_pred"] == 'yes'].index
        test_data.data_pd.loc[yes_ids, 'yes_vote'] += 1

        no_ids = test_data.data_pd[test_data.data_pd.loc[:, "current_pred"] == 'no'].index
        test_data.data_pd.loc[no_ids, 'no_vote'] += 1

    x_train = training_data.csr_mat
    y_train = training_data.data_pd.loc[:, 'label']

    sss = StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=0)
    for train_index, tune_index in sss.split(x_train, y_train):
        x_train_flash, x_tune_flash = x_train[train_index], x_train[tune_index]
        y_train_flash, y_tune_flash = y_train.iloc[train_index], y_train.iloc[tune_index]
        #best_conf = tune_with_flash(x_train_flash, y_train_flash, x_tune_flash, y_tune_flash, fold_num=1,
        #                            pool_size=100000, init_pool=10, budget=10, label=project_name)
        print(best_conf)
    clf = SVC(C=BEST_CONF_TEMP[0], kernel=BEST_CONF_TEMP[1], gamma=BEST_CONF_TEMP[2], coef0=BEST_CONF_TEMP[3],
              probability=True, random_state=0)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(test_data.csr_mat)
    decision_dist = clf.decision_function(test_data.csr_mat)
    y_pred_proba = clf.predict_proba(test_data.csr_mat)[:, 1]
    test_data.data_pd.loc[:, 'total_dataset'] = y_pred.tolist()
    test_data.data_pd.loc[:, 'distance_from_plane'] = decision_dist.tolist()
    test_data.data_pd.loc[:, 'probability'] = y_pred_proba.tolist()

    vote_results(test_data)
    print(project_name + " | SINGLE F Score: ")
    print(get_report(test_data, 'total_dataset'))
    print(project_name + " | ENSEMBLE F Score: ")
    print(get_report(test_data, 'code_ensemble'))
    print(project_name + " | COMBINED PROBABILITY Delta " + str(PROBA_DELTA) + " F Score: ")
    print(get_report(test_data, 'code_combined'))
    print(project_name + " | COMBINED DISTANCE Delta " + str(DELTA) + " F Score: ")
    print(get_report(test_data, 'code'))

    #(12.234889648250906, 'rbf', 0.8249527472853467, 0.27927738511824773)


def vote_results(test_data, uncertain_dist=DELTA):
    for i, row in test_data.data_pd.iterrows():
        if(abs(row['distance_from_plane'])>DELTA):
            test_data.data_pd.at[i, 'code'] = row['total_dataset']
        else:
            if row['yes_vote']>row['no_vote']:
                test_data.data_pd.at[i, 'code'] = 'yes'
            else:
                test_data.data_pd.at[i, 'code'] = 'no'

        if row['yes_vote']>row['no_vote']:
            test_data.data_pd.at[i, 'code_ensemble'] = 'yes'
        else:
            test_data.data_pd.at[i, 'code_ensemble'] = 'no'

        if (abs(row['probability'] - .5) > PROBA_DELTA):
            test_data.data_pd.at[i, 'code_combined'] = row['total_dataset']
        else:
            if row['yes_vote'] > row['no_vote']:
                test_data.data_pd.at[i, 'code_combined'] = 'yes'
            else:
                test_data.data_pd.at[i, 'code_combined'] = 'no'

def get_report(test_data, field):
    y_pred = test_data.data_pd.loc[:, field]
    y_test = test_data.data_pd.loc[:, 'label']
    return classification_report(y_test,y_pred)



#NEW TESTING

def classify(training_data, test_data, project_name, all_datasets):
    import warnings
    warnings.filterwarnings("ignore", category=SettingWithCopyWarning)


    y_test = test_data.data_pd.loc[:, 'label']
    test_data.data_pd.loc[:, 'yes_vote'] = 0
    test_data.data_pd.loc[:, 'no_vote'] = 0

    x_train = training_data.csr_mat
    y_train = training_data.data_pd.loc[:, 'label']

    # SVM Test
    sss = StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=0)
    for train_index, tune_index in sss.split(x_train, y_train):
        x_train_flash, x_tune_flash = x_train[train_index], x_train[tune_index]
        y_train_flash, y_tune_flash = y_train.iloc[train_index], y_train.iloc[tune_index]
        best_conf_svm = tune_with_flash(x_train_flash, y_train_flash, x_tune_flash, y_tune_flash, fold_num=1,
                                   pool_size=10000, init_pool=10, budget=10, label=project_name)
    clf = SVC(C=best_conf_svm[0], kernel=best_conf_svm[1], gamma=best_conf_svm[2], coef0=best_conf_svm[3],
              probability=True, random_state=0)
    predict(clf, x_train, y_train, test_data, 'svm_pred', project_name)

    # NBM Test
    clf = MultinomialNB(alpha=1.0)
    predict(clf, x_train, y_train, test_data, 'nbm_pred', project_name)

    # KNN Test
    for train_index, tune_index in sss.split(x_train, y_train):
        x_train_knn, x_tune_knn = x_train[train_index], x_train[tune_index]
        y_train_knn, y_tune_knn = y_train.iloc[train_index], y_train.iloc[tune_index]

        best_accu = 0
        best_k = 1
        for i, k in enumerate(np.arange(1,9)):
            # Setup a knn classifier with k neighbors
            clf = KNeighborsClassifier(n_neighbors=k)

            # Fit the model
            clf.fit(x_train_knn, y_train_knn)

            accu = clf.score(x_tune_knn, y_tune_knn)
            if best_accu < accu:
                best_k = k
                best_accu = accu

    clf = KNeighborsClassifier(n_neighbors=best_k, weights='uniform', algorithm='auto', leaf_size=30, p=2)
    predict(clf, x_train, y_train, test_data, 'knn_pred', project_name)

    # This part is just for understanding the prev work
    # trying sub classifier of NBM
    test_data.data_pd.loc[:, 'yes_vote'] = 0
    test_data.data_pd.loc[:, 'no_vote'] = 0
    for train_dataset_name in all_datasets:
        if train_dataset_name in project_name:
            continue
        training_ids = training_data.data_pd.loc[training_data.data_pd['projectname'] == train_dataset_name]
        training_ids = list(training_ids.index)

        x_train = training_data.csr_mat[training_ids]
        y_train = training_data.data_pd.loc[training_ids, 'label']

        clf = MultinomialNB(alpha=1.0)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(test_data.csr_mat)

        test_data.data_pd.loc[:, 'nbm_sub_pred'] = y_pred.tolist()

        yes_ids = test_data.data_pd[test_data.data_pd.loc[:, "nbm_sub_pred"] == 'yes'].index
        test_data.data_pd.loc[yes_ids, 'yes_vote'] += 1

        no_ids = test_data.data_pd[test_data.data_pd.loc[:, "nbm_sub_pred"] == 'no'].index
        test_data.data_pd.loc[no_ids, 'no_vote'] += 1

    for i, row in test_data.data_pd.iterrows():
        if row['yes_vote']>row['no_vote']:
            test_data.data_pd.at[i, 'nbm_sub_pred'] = 'yes'
        else:
            test_data.data_pd.at[i, 'nbm_sub_pred'] = 'no'

    print(project_name + " | nbm_sub_pred" + "\n" + get_report(test_data, 'nbm_sub_pred'))
    # END OF SUB CLF

    test_data.data_pd.to_csv(project_name + '_data_pd.csv')


def predict(clf, x_train, y_train, test_data, col_name, project_name):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(test_data.csr_mat)
    y_pred_proba = clf.predict_proba(test_data.csr_mat)[:, 1]
    test_data.data_pd.loc[:, col_name] = y_pred.tolist()
    test_data.data_pd.loc[:, col_name + '_proba'] = y_pred_proba.tolist()

    print(project_name + " | " + col_name + "\n" + get_report(test_data, col_name))

def predict_active(clf, x_train, y_train, test_data, col_name, project_name):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(test_data.csr_mat)
    y_pred_proba = clf.predict_proba(test_data.csr_mat)[:, 1]
    test_data.data_pd.loc[:, col_name] = y_pred.tolist()
    test_data.data_pd.loc[:, col_name + '_proba'] = y_pred_proba.tolist()

    print(project_name + " | " + col_name + "\n" + get_report(test_data, col_name))

    neg_ids = np.where(test_data.data_pd.svm_pred_proba < .1)[0]
    pos_ids = np.where(test_data.data_pd.svm_pred_proba > .9)[0]

    new_pos_mat = test_data.csr_mat[pos_ids]
    new_neg_mat = test_data.csr_mat[neg_ids]
    new_pos_y = test_data.data_pd.loc[pos_ids, 'label']
    new_neg_y = test_data.data_pd.loc[neg_ids, 'label']

    x = sp.vstack([x_train, new_pos_mat])
    x = sp.vstack([x, new_neg_mat])


