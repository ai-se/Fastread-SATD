from sklearn import clone
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC



def predict_svm(mar, C=1.0, kernel='rbf', gamma='auto', coef0=0.0):
    # Stratification attempts to maintain the realtive ratios
    # of positive and negative classes in each of folds.
    # This is very helpful for skewed datasets (read: SE Data)
    logger.info("C: " + str(C) + " kernel: " + kernel + " gamma: " + str(gamma) + " coef0: " + str(coef0))

    skfolds = StratifiedKFold(n_splits=2, random_state=0)

    # Create and instance of your classifier
    clf = SVC(C=C, kernel=kernel, gamma=gamma, degree=3, coef0=0.0)

    # Store F1, precision, and recall for svm
    svm_f1 = []
    svm_prec = []
    svm_recl = []

    fold = 0

    test = mar.body['label']

    for train_index, test_index in skfolds.split(mar.csr_mat, test):
        # Deep copy here, when you do cross-validation, it's always a good idead
        # not to mess with the original classifier.

        # We need a "clean" classifier for every fold. Otherwise your mixing the
        # training and testing data form different folds. That's a no-no.
        cloned_clf = clone(clf)

        # Training data
        # -------------
        X_train_folds = mar.csr_mat[train_index]
        y_train_folds = test[train_index]

        # Testing data
        # ------------
        X_test_folds = mar.csr_mat[test_index]
        y_test_folds = test[test_index]

        # Fit a classifier on the training data
        # -------------------------------------
        cloned_clf.fit(X_train_folds, y_train_folds)

        # Make predictions on a test set
        # ------------------------------
        y_hat = cloned_clf.predict(X_test_folds)

        # Compute some metrics here. Like Precision, Recall, False Alarm, or what have you.
        # ---------------------------------------------------------------------------------
        cmat = confusion_matrix(y_test_folds, y_hat)

        # Precision
        # ---------
        prec = cmat[1, 1] / (cmat[1, 1] + cmat[0, 1])

        # Recall
        # ------
        recall = cmat[1, 1] / (cmat[1, 1] + cmat[1, 0])

        # F1 Score
        # --------
        f1 = 2 * (prec * recall) / (prec + recall)

        # Record results
        # --------------

        svm_f1.append(f1)
        svm_prec.append(prec)
        svm_recl.append(recall)

        print("split done")

    f1_mean = sum(svm_f1) / len(svm_f1)
    prec_mean = sum(svm_prec) / len(svm_prec)
    recl_mean = sum(svm_recl) / len(svm_recl)



