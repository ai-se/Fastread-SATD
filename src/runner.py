from __future__ import division, print_function
import numpy as np
from pdb import set_trace

from demos import cmd
import pickle
import matplotlib.pyplot as plt
from mar import MAR
from sk import rdivDemo
import random
from collections import Counter

import pandas
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix

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


if __name__ == "__main__":

    active_learning('td_2016.csv')
    print("END")
    # eval(cmd())


    # TODO implement BM25