import pickle
import random
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

import pandas

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.disabled = True

class MAR(object):
    def __init__(self):
        self.fea_count = 4000       # FAHID feature counts for building TFIDF
        self.step = 10              # FAHID after how many steps we want to train again
        self.enough = 30            # FAHID convert to agressive undersampling
        self.atleast = 100          # FAHID if we have unlabeled data, assume all negative (as the chances are very low)
        self.interval = 500000000   # FAHID should we do error correction or not, or something to do with that
        random.seed(0)

        # FAHID
        self.true_count = 0
        self.false_count = 0

    def create(self, filename, dataset=None):
        self.filename = filename
        self.hasLabel = True
        self.body = None
        self.round = 0
        self.dataset = dataset

        try:
            self.loadfile()
            self.preprocess()
        except:
            ## cannot find file in workspace ##
            print("Data file not found")
        return self

    def loadfile(self):
        with open("../workspace/td/" + self.filename, "r") as csvfile:
            content = [x for x in csv.reader(csvfile, delimiter=',')]

        if self.dataset:
            project_based_content = [x for x in content[1:] if x[0]==self.dataset]
        else:
            project_based_content = content

        data_pd = pandas.DataFrame(project_based_content[1:], columns=content[0])

        data_pd['label'] = np.where((data_pd['classification'] != "WITHOUT_CLASSIFICATION"), 'yes', 'no')
        data_pd['code'] = 'undetermined'
        data_pd['time'] = 0
        data_pd['fixed'] = 0
        data_pd['count'] = 0
        true_count = len(data_pd[(data_pd['label'] == 'yes')])
        false_count = len(data_pd[(data_pd['label'] == 'no')])
        print("Ground Truth: True " + str(true_count) + " | False " + str(false_count))

        self.true_count = true_count
        self.false_count = false_count

        self.body = data_pd
        return

    # FAHID predicted results
    def get_numbers(self):
        total = len(self.body["code"])
        pos = len(self.body[self.body["code"] == 'yes'])
        neg = len(self.body[self.body["code"] == 'no'])

        self.pool = np.where(np.array(self.body['code']) == "undetermined")[0]
        self.labeled = list(set(range(len(self.body['code']))) - set(self.pool))
        return pos, neg, total

    def preprocess(self):
        ### Feature selection by tfidf in order to keep vocabulary ###
        tfidfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=None, use_idf=True, smooth_idf=False,
                                  sublinear_tf=False, decode_error="ignore", max_features=self.fea_count)
        tfidfer.fit(self.body['commenttext'])

        self.voc = list(tfidfer.vocabulary_.keys())

        ### Term frequency as feature, L2 normalization ##########
        tfer = TfidfVectorizer(lowercase=True, stop_words="english", norm='l2', use_idf=False,
                               vocabulary=self.voc, decode_error="ignore")

        self.csr_mat = tfer.fit_transform(self.body['commenttext'])
        return

    ## FAHID TODO GOING TO REMOVE ##
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
    ## FAHID TODO END ##

    def get_random_ids(self):
        return self.body.sample(self.step).index

    ## Train model ##
    def train(self, pne=True, weighting=True):
        clf = svm.SVC(kernel='linear', probability=True, class_weight='balanced') if weighting else svm.SVC(
            kernel='linear', probability=True)

        poses = self.body.loc[self.body['code'] == 'yes']
        negs = self.body.loc[self.body['code'] == 'no']
        labeled = self.body.loc[self.body['code'] != 'undetermined']
        unlabeled = self.body.loc[self.body['code'] == 'undetermined']

        pos_ids = list(poses.index)
        labeled_ids = list(labeled.index)

        try:
            unlabeled = unlabeled.sample(self.atleast)
        except:
            pass

        # TODO FAHID PRESUMTIVE NON RELEVANT AFTER APPLYING BM25
        # Examples Presume all examples are false, because true examples are few
        # This reduces the biasness of not doing random sampling
        if not pne:
            unlabeled = []

        unlabeled_ids = unlabeled.index

        labels = np.array([x if x != 'undetermined' else 'no' for x in self.body['code']])
        all_neg = pandas.concat([negs, unlabeled])
        all_neg_ids = list(all_neg.index)

        sample = pandas.concat([poses, negs, unlabeled])
        sample_ids = list(sample.index)

        start = time.time()

        clf.fit(self.csr_mat[sample_ids], labels[sample_ids])

        logger.info("after fitting: " + str(time.time() - start))

        ## aggressive undersampling ##
        if len(poses) >= self.enough:
            start = time.time()

            train_dist = clf.decision_function(self.csr_mat[all_neg_ids])
            pos_at = list(clf.classes_).index("yes")
            if pos_at:
                train_dist = -train_dist
            negs_sel = np.argsort(train_dist)[::-1][:len(pos_ids)]
            sample_ids = list(pos_ids) + list(np.array(all_neg_ids)[negs_sel])

            logger.info("before fitting > enough: " + str(time.time() - start))
            start = time.time()

            clf.fit(self.csr_mat[sample_ids], labels[sample_ids])

            logger.info("after fitting > enough: " + str(time.time() - start))
        elif pne:
            start = time.time()

            train_dist = clf.decision_function(self.csr_mat[unlabeled_ids])
            pos_at = list(clf.classes_).index("yes")
            if pos_at:
                train_dist = -train_dist
            unlabel_sel = np.argsort(train_dist)[::-1][:int(len(unlabeled_ids) / 2)]
            sample_ids = list(labeled_ids) + list(np.array(unlabeled_ids)[unlabel_sel])

            logger.info("before fitting pne: " + str(time.time() - start))
            start = time.time()

            clf.fit(self.csr_mat[sample_ids], labels[sample_ids])

            logger.info("after fitting pne: " + str(time.time() - start))

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
        self.body.loc[id, 'code'] = label
        self.body.loc[id, 'time'] = time.time()

    def code_error(self, id, error='none'):
        # FAHID: simulate a human reader
        if error == 'random':
            self.code_random(id, self.body.loc[id, 'label'])
        elif error == 'three':
            self.code_three(id, self.body['label'][id])
        else:
            self.code(id, self.body.loc[id, 'label'])

    def code_three(self, id, label):
        self.code_random(id, label)
        self.code_random(id, label)
        if self.body['fixed'][id] == 0:
            self.code_random(id, label)

    def code_random(self, id, label):
        error_rate = 0.3
        if label == 'yes':
            if random.random() < error_rate:
                new = 'no'
            else:
                new = 'yes'
        else:
            if random.random() < error_rate :
                new = 'yes'
            else:
                new = 'no'
        if new == self.body.loc[id, "code"]:
            self.body.loc[id, 'fixed'] = 1
        self.body.loc[id, "code"] = new
        self.body.loc[id, "time"] = time.time()
        self.body.loc[id, "count"] = self.body.loc[id, "count"] + 1


    def get_allpos(self):
        return len(self.body[self.body['label'] == 'yes'])


