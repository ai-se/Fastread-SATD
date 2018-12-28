import numpy as np
STEP_SIZE = 10

def fastread(clf, test_data, x_train_folds, y_train_folds):

    clf.fit(x_train_folds, y_train_folds)

    current_pos = 0
    target = len(test_data.data_pd.loc[test_data.data_pd['label'] == "yes"])
    #TODO find a stopping rule, now using total target

    while current_pos<target:
        current_pos = len(test_data.data_pd.loc[test_data.data_pd['code'] == "yes"])
        pool = np.where(np.array(test_data.data_pd['code']) == "undetermined")[0]

        uncertain_id, uncertain_prob = uncertain(clf, test_data, pool)
        certain_id, certain_prob = certain(clf, test_data, pool)

        if current_pos < 50:
            # Uncertainity Sampling
            for id in uncertain_id:
                code_error(id, test_data)
        else:
            break
            # Certainity Sampling
            for id in certain_id:
                code_error(id, test_data)

    all_yes_pos = np.argwhere(test_data.data_pd['code'] == "yes")

    return uncertain_id, uncertain_prob, certain_id, certain_prob


## Get certain ##
def certain(clf, test_data, pool):
    pos_at = list(clf.classes_).index("yes")
    prob = clf.predict_proba(test_data.csr_mat[pool])[:, pos_at]
    order = np.argsort(prob)[::-1][:STEP_SIZE]
    return np.array(pool)[order], np.array(prob)[order]

## Get uncertain ##
def uncertain(clf, test_data, pool):
    pos_at = list(clf.classes_).index("yes")
    a = clf.predict_proba(test_data.csr_mat[pool])
    prob = a[:, pos_at]
    train_dist = clf.decision_function(test_data.csr_mat[pool])
    order = np.argsort(np.abs(train_dist))[:STEP_SIZE]  ## uncertainty sampling by distance to decision plane
    return np.array(pool)[order], np.array(prob)[order]


def code_error(id, test_data, error=None):
    a = test_data.data_pd.iloc[id]['label']
    test_data.data_pd.at[id, 'code'] = a