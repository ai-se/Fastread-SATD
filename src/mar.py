import pickle
from pdb import set_trace
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
from collections import Counter
from sklearn import svm
import matplotlib.pyplot as plt
import time
import os
from sklearn import preprocessing


class MAR(object):
    def __init__(self):
        self.step = 10              # FAHID after how many steps we want to train again
        self.enough = 30            # FAHID convert to agressive undersampling
        self.atleast = 100          # FAHID if we have unlabeled data, assume all negative (as the chances are very low)
        self.interval = 500000000   # FAHID should we do error correction or not, or something to do with that

    def create(self, filename):
        self.filename = filename
        self.hasLabel = True
        self.body = {}
        self.round = 0

        try:
            self.loadfile()
            self.preprocess()
        except:
            ## cannot find file in workspace ##
            print("Data file not found")
        return self


    def loadfile(self):
        with open("../workspace/data/" + str(self.filename), "r") as csvfile:
            content = [x for x in csv.reader(csvfile, delimiter=',')]
        fields = ["Document Title", "Abstract", "Year", "PDF Link"]
        header = content[0]
        for field in fields:
            ind = header.index(field)
            self.body[field] = [c[ind] for c in content[1:]]
        try:
            ind = header.index("label")
            self.body["label"] = [c[ind] for c in content[1:]]
        except:
            self.hasLabel = False
            self.body["label"] = ["unknown"] * (len(content) - 1)
        try:
            ind = header.index("code")
            self.body["code"] = [c[ind] for c in content[1:]]
        except:
            self.body["code"] = ['undetermined'] * (len(content) - 1)
        try:
            ind = header.index("time")
            self.body["time"] = [c[ind] for c in content[1:]]
        except:
            self.body["time"] = [0] * (len(content) - 1)
        try:
            ind = header.index("syn_error")
            self.body["syn_error"] = [c[ind] for c in content[1:]]
        except:
            self.body["syn_error"] = [0] * (len(content) - 1)
        try:
            ind = header.index("fixed")
            self.body["fixed"] = [c[ind] for c in content[1:]]
        except:
            self.body["fixed"] = [0] * (len(content) - 1)
        try:
            ind = header.index("count")
            self.body["count"] = [c[ind] for c in content[1:]]
        except:
            self.body["count"] = [0] * (len(content) - 1)
        return

    # FAHID predicted results
    def get_numbers(self):
        total = len(self.body["code"])
        pos = Counter(self.body["code"])["yes"]
        neg = Counter(self.body["code"])["no"]

        self.pool = np.where(np.array(self.body['code']) == "undetermined")[0]
        self.labeled = list(set(range(len(self.body['code']))) - set(self.pool))
        return pos, neg, total


    def preprocess(self):
        ### Combine title and abstract for training ###########
        content = [self.body["Document Title"][index] + " " + self.body["Abstract"][index] for index in
                   range(len(self.body["Document Title"]))]
        #######################################################

        ### Feature selection by tfidf in order to keep vocabulary ###
        tfidfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=None, use_idf=True, smooth_idf=False,
                                  sublinear_tf=False, decode_error="ignore", max_features=4000)
        tfidfer.fit(content)
        self.voc = list(tfidfer.vocabulary_.keys())

        ##############################################################

        ### Term frequency as feature, L2 normalization ##########
        tfer = TfidfVectorizer(lowercase=True, stop_words="english", norm='l2', use_idf=False,
                               vocabulary=self.voc, decode_error="ignore")
        # tfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=None, use_idf=False,
        #                 vocabulary=self.voc,decode_error="ignore")
        self.csr_mat = tfer.fit_transform(content)
        ########################################################
        return


    ## BM25 ##
    def BM25(self, query):
        if query[0] == '':
            # FAHID This is not a random choice, but to generate n random numbers between [0,1)
            test = len(self.body["Document Title"])
            self.bm = np.random.rand(len(self.body["Document Title"]))
            return

        b = 0.75
        k1 = 1.5

        ### Combine title and abstract for training ###########
        content = [self.body["Document Title"][index] + " " + self.body["Abstract"][index] for index in
                   range(len(self.body["Document Title"]))]
        #######################################################

        ### Feature selection by tfidf in order to keep vocabulary ###

        tfidfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=None, use_idf=False, smooth_idf=False,
                                  sublinear_tf=False, decode_error="ignore")
        tf = tfidfer.fit_transform(content)
        d_avg = np.mean(np.sum(tf, axis=1))
        score = {}
        for word in query:
            score[word] = []
            try:
                id = tfidfer.vocabulary_[word]
            except:
                score[word] = [0] * len(content)
                continue
            df = sum([1 for wc in tf[:, id] if wc > 0])
            idf = np.log((len(content) - df + 0.5) / (df + 0.5))
            for i in range(len(content)):
                score[word].append(
                    idf * tf[i, id] / (tf[i, id] + k1 * ((1 - b) + b * np.sum(tf[0], axis=1)[0, 0] / d_avg)))
        self.bm = np.sum(list(score.values()), axis=0)

    def BM25_get(self):
        # FAHID: get the indexes of bm at indexes of pool, then reverse and take the first step size of them
        return self.pool[np.argsort(self.bm[self.pool])[::-1][:self.step]]

    ## Train model ##
    def train(self, pne=True, weighting=True):

        clf = svm.SVC(kernel='linear', probability=True, class_weight='balanced') if weighting else svm.SVC(
            kernel='linear', probability=True)
        poses = np.where(np.array(self.body['code']) == "yes")[0]
        negs = np.where(np.array(self.body['code']) == "no")[0]
        left = poses
        decayed = list(left) + list(negs)
        unlabeled = np.where(np.array(self.body['code']) == "undetermined")[0]
        try:
            unlabeled = np.random.choice(unlabeled, size=np.max((len(left), self.atleast)), replace=False)
        except:
            pass

        if not pne:
            unlabeled = []

        labels = np.array([x if x != 'undetermined' else 'no' for x in self.body['code']])
        all_neg = list(negs) + list(unlabeled)
        sample = list(decayed) + list(unlabeled)

        clf.fit(self.csr_mat[sample], labels[sample])

        ## aggressive undersampling ##
        if len(poses) >= self.enough:

            train_dist = clf.decision_function(self.csr_mat[all_neg])
            pos_at = list(clf.classes_).index("yes")
            if pos_at:
                train_dist = -train_dist
            negs_sel = np.argsort(train_dist)[::-1][:len(left)]
            sample = list(left) + list(np.array(all_neg)[negs_sel])
            clf.fit(self.csr_mat[sample], labels[sample])
        elif pne:
            train_dist = clf.decision_function(self.csr_mat[unlabeled])
            pos_at = list(clf.classes_).index("yes")
            if pos_at:
                train_dist = -train_dist
            unlabel_sel = np.argsort(train_dist)[::-1][:int(len(unlabeled) / 2)]
            sample = list(decayed) + list(np.array(unlabeled)[unlabel_sel])
            clf.fit(self.csr_mat[sample], labels[sample])

        uncertain_id, uncertain_prob = self.uncertain(clf)
        certain_id, certain_prob = self.certain(clf)

        return uncertain_id, uncertain_prob, certain_id, certain_prob


    ## Get certain ##
    def certain(self, clf):
        pos_at = list(clf.classes_).index("yes")
        prob = clf.predict_proba(self.csr_mat[self.pool])[:, pos_at]
        order = np.argsort(prob)[::-1][:self.step]
        return np.array(self.pool)[order], np.array(prob)[order]

    ## Get uncertain ##
    def uncertain(self, clf):
        pos_at = list(clf.classes_).index("yes")
        prob = clf.predict_proba(self.csr_mat[self.pool])[:, pos_at]
        train_dist = clf.decision_function(self.csr_mat[self.pool])
        order = np.argsort(np.abs(train_dist))[:self.step]  ## uncertainty sampling by distance to decision plane
        # order = np.argsort(np.abs(prob-0.5))[:self.step]    ## uncertainty sampling by prediction probability
        return np.array(self.pool)[order], np.array(prob)[order]

    ## Get random ##
    def random(self):
        return np.random.choice(self.pool, size=np.min((self.step, len(self.pool))), replace=False)

    ## Get one random ##
    def one_rand(self):
        pool_yes = [x for x in range(len(self.body['label'])) if self.body['label'][x] == 'yes']
        return np.random.choice(pool_yes, size=1, replace=False)


    ## Code candidate studies ##
    def code(self, id, label):
        self.body["code"][id] = label
        self.body["time"][id] = time.time()

    def code_error(self, id, error='none'):
        # FAHID: simulate a human reader
        if error == 'circle':
            self.code_circle(id, self.body['label'][id])
        elif error == 'random':
            self.code_random(id, self.body['label'][id])
        elif error == 'three':
            self.code_three(id, self.body['label'][id])
        else:
            self.code(id, self.body['label'][id])


    def code_three(self, id, label):
        self.code_random(id, label)
        self.code_random(id, label)
        if self.body['fixed'][id] == 0:
            self.code_random(id, label)


    def code_random(self, id, label):
        import random
        error_rate = 0.3
        if label == 'yes':
            if random.random() < error_rate:
                new = 'no'
            else:
                new = 'yes'
        else:
            if random.random() < error_rate:
                new = 'yes'
            else:
                new = 'no'
        if new == self.body["code"][id]:
            self.body['fixed'][id] = 1
        self.body["code"][id] = new
        self.body["time"][id] = time.time()
        self.body["count"][id] = self.body["count"][id] + 1


    def get_allpos(self):
        return len([1 for c in self.body["label"] if c == "yes"])

