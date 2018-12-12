from __future__ import division, print_function
import numpy as np
from pdb import set_trace

from demos import cmd
import pickle
import matplotlib.pyplot as plt

from duo import predict_svm
from loader import SATDD
from mar import MAR
from processor import k_fold_with_tuning
from rig import run_rig
from sk import rdivDemo
import random
from collections import Counter

import pandas
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import numpy as np

# FAHID NOTE I dont think we need to change anything here for the new dataset
def active_learning(filename, query='', stopat=.95, error='three', interval=100000, seed=0):
    stopat = float(stopat)                  # FAHID stop at recall
    starting = 5                            # FAHID Do random sampling until
    np.random.seed(seed)

    read = MAR()
    read = read.create(filename)

    #read.BM25(query.strip().split('_'))     # FAHID TODO REMOVE

    num_of_total_pos = read.get_allpos()
    target = int(num_of_total_pos * stopat)

    print ("Target: " + str(target))

    while True:
        pos, neg, total = read.get_numbers()
        # print("%d, %d" % (pos, pos + neg))


        if pos + neg >= total:
            break

        if pos < starting:
            for id in read.get_random_ids():
                read.code_error(id, error=error)
        else:
            a, b, c, d = read.train(weighting=True, pne=True)
            if pos >= target:
                break

            # QUERY
            if pos < 10:
                # Uncertainity Sampling
                for id in a:
                    read.code_error(id, error=error)
            else:
                # Certainity Sampling
                for id in c:
                    read.code_error(id, error=error)

    pos, neg, total = read.get_numbers()
    print("Positive %d, Total Looked %d" % (pos, pos + neg))

    checked_data = read.body.loc[read.body["code"]!='undetermined']
    test = checked_data.loc[:, "label"].tolist()
    predicted = checked_data.loc[:, "code"].tolist()
    confusion_mat = confusion_matrix(test, predicted, labels=["no", "yes"])
    score = metrics.classification_report(test, predicted, digits=3)

    print(score)
    print(confusion_mat)
    # #set_trace()
    # return read




def duo(filename):
    satdd = SATDD()
    satdd = satdd.load_data(filename)
    training_data = satdd.create_and_process_dataset(['apache-ant-1.7.0', 'apache-jmeter-2.10', 'columba-1.4-src'],
                                                     doInclude=False)
    # no need to give a tfidf, will calculate itself
    training_data.set_csr_mat()
    test_data = satdd.create_and_process_dataset(['columba-1.4-src'], doInclude=True)
    # need to give the tfidf from training set, will just use transform to create csr_matrix
    test_data.set_csr_mat(training_data.tfer)

    conf_mat = k_fold_with_tuning(test_data, training_data)
    print("END")

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
    logger = logging.getLogger(__name__)

    run_rig('td_2016.csv')
    # active_learning('td_2016.csv')


    # duo('td_2016.csv')
    # eval(cmd())
