import numpy as np
from pandas.core.common import SettingWithCopyWarning
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
import logging

from processor import calc_f, tune_with_flash

logger = logging.getLogger(__name__)

BEST_CONF_TEMP = [47.96, 'poly', 0.84, 0.75]

def ensemble_vote(training_data, test_data, project_name, all_datasets):
    import warnings
    warnings.filterwarnings("ignore", category=SettingWithCopyWarning)


    y_test = test_data.data_pd.loc[:, 'label']
    test_data.data_pd.loc[:, 'yes_vote'] = 0
    test_data.data_pd.loc[:, 'no_vote'] = 0

    test_data.data_pd.loc[:, 'total_dataset'] = ''
    test_data.data_pd.loc[:, 'probability'] = 0

    ids = 0

    for train_dataset_name in all_datasets:
        ids += 1
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
            print(train_dataset_name + " | ")
            print(best_conf)
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
        best_conf = tune_with_flash(x_train_flash, y_train_flash, x_tune_flash, y_tune_flash, fold_num=1,
                                    pool_size=100000,
                                    init_pool=10, budget=10, label=project_name)
        print("TOTAL | ")
        print(best_conf)
    clf = SVC(C=best_conf[0], kernel=best_conf[1], gamma=best_conf[2], coef0=best_conf[3],
              probability=True, random_state=0)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(test_data.csr_mat)
    decision_dist = clf.decision_function(test_data.csr_mat)
    test_data.data_pd.loc[:, 'total_dataset'] = y_pred.tolist()
    test_data.data_pd.loc[:, 'probability'] = decision_dist.tolist()
    print("HI")

    #(12.234889648250906, 'rbf', 0.8249527472853467, 0.27927738511824773)

