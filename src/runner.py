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


def active_learning(filename, query='', stopat=.95, error='none', interval=100000, seed=0):
    stopat = float(stopat)                  # FAHID stop at recall
    starting = 1                            # FAHID Do random sampling until
    np.random.seed(seed)

    read = MAR()
    read = read.create(filename)

    read.interval = interval                # FAHID LATER TRAIN USE IT correct errors with human-machine disagreements

    read.BM25(query.strip().split('_'))     # FAHID LATER

    num2 = read.get_allpos()
    target = int(num2 * stopat)

    while True:
        pos, neg, total = read.get_numbers()
        print("%d, %d" % (pos, pos + neg))

        if pos + neg >= total:
            break

        if pos < starting:
            # FAHID TODO LETS RANDOMLY PICK self.step undetermined values
            for id in read.BM25_get():
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
    set_trace()
    return read


if __name__ == "__main__":
    active_learning('Hall.csv')
    # eval(cmd())
