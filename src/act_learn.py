import numpy as np
STEP_SIZE = 10

def active_learning(clf, test_dataset):

    current_val = 0
    target = y_test

    while current_val<target:
        pool = np.where(np.array(self.body['code']) == "undetermined")[0]

        uncertain_id, uncertain_prob = uncertain(clf, x_test, pool)
        certain_id, certain_prob = certain(clf, x_test, pool)

        if pos < 10:
            # Uncertainity Sampling
            for id in a:
                read.code_error(id, error=error)
        else:
            # Certainity Sampling
            for id in c:
                read.code_error(id, error=error)

    return uncertain_id, uncertain_prob, certain_id, certain_prob


## Get certain ##
def certain(clf, x_test, pool):
    pos_at = list(clf.classes_).index("yes")
    prob = clf.predict_proba(x_test[pool])[:, pos_at]
    order = np.argsort(prob)[::-1][:STEP_SIZE]
    return np.array(pool)[order], np.array(prob)[order]

## Get uncertain ##
def uncertain(clf, x_test, pool):
    pos_at = list(clf.classes_).index("yes")
    prob = clf.predict_proba(x_test[pool])[:, pos_at]
    train_dist = clf.decision_function(x_test[pool])
    order = np.argsort(np.abs(train_dist))[:STEP_SIZE]  ## uncertainty sampling by distance to decision plane
    return np.array(pool)[order], np.array(prob)[order]
