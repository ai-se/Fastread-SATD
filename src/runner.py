from __future__ import division, print_function

import numpy as np
from pdb import set_trace
from demos import cmd
import pickle
import matplotlib.pyplot as plt

from sk import rdivDemo

import random

from collections import Counter

def active_learning(filename, query='', stop='true', stopat=1.00, error='none', interval = 100000, seed=0):
    stopat = float(stopat)
    thres = 0
    starting = 1
    counter = 0
    pos_last = 0
    np.random.seed(seed)

    read = MAR()
    read = read.create(filename)

    read.interval = interval

    read.BM25(query.strip().split('_'))


    num2 = read.get_allpos()
    target = int(num2 * stopat)
    if stop == 'est':
        read.enable_est = True
    else:
        read.enable_est = False

    while True:
        pos, neg, total = read.get_numbers()
        try:
            print("%d, %d, %d" %(pos,pos+neg, read.est_num))
        except:
            print("%d, %d" %(pos,pos+neg))

        if pos + neg >= total:
            if stop=='knee' and error=='random':
                coded = np.where(np.array(read.body['code']) != "undetermined")[0]
                seq = coded[np.argsort(read.body['time'][coded])]
                part1 = set(seq[:read.kneepoint * read.step]) & set(
                    np.where(np.array(read.body['code']) == "no")[0])
                part2 = set(seq[read.kneepoint * read.step:]) & set(
                    np.where(np.array(read.body['code']) == "yes")[0])
                for id in part1 | part2:
                    read.code_error(id, error=error)
            break

        if pos < starting or pos+neg<thres:
            for id in read.BM25_get():
                read.code_error(id, error=error)
        else:
            a,b,c,d =read.train(weighting=True,pne=True)
            if stop == 'est':
                if stopat * read.est_num <= pos:
                    break
            elif stop == 'soft':
                if pos>=10 and pos_last==pos:
                    counter = counter+1
                else:
                    counter=0
                pos_last=pos
                if counter >=5:
                    break
            elif stop == 'knee':
                if pos>=10:
                    if read.knee():
                        if error=='random':
                            coded = np.where(np.array(read.body['code']) != "undetermined")[0]
                            seq = coded[np.argsort(np.array(read.body['time'])[coded])]
                            part1 = set(seq[:read.kneepoint * read.step]) & set(
                                np.where(np.array(read.body['code']) == "no")[0])
                            part2 = set(seq[read.kneepoint * read.step:]) & set(
                                np.where(np.array(read.body['code']) == "yes")[0])
                            for id in part1|part2:
                                read.code_error(id, error=error)
                        break
            else:
                if pos >= target:
                    break
            if pos < 10:
                for id in a:
                    read.code_error(id, error=error)
            else:
                for id in c:
                    read.code_error(id, error=error)
    set_trace()                
    return read


if __name__ == "__main__":
    eval(cmd())
