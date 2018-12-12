import csv
import logging
import string

import pandas
import random
import numpy as np
from nltk import WordNetLemmatizer, sent_tokenize, wordpunct_tokenize, pos_tag
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from utility import tokenize, top_mean_feats

import logging
logger = logging.getLogger(__name__)


class SATDD:
    def __init__(self):
        self.all_dataset_pd = None
        self.data_pds = []
        self.total_true = 0
        self.total_false = 0

        random.seed(0)

        self.filename = None

    def load_data(self, filename):
        self.filename = filename
        with open("../workspace/td/" + self.filename, "r") as csvfile:
            content = [x for x in csv.reader(csvfile, delimiter=',')]

        self.all_dataset_pd = pandas.DataFrame(content[1:], columns=content[0])

        self.all_dataset_pd['label'] = np.where((self.all_dataset_pd['classification'] != "WITHOUT_CLASSIFICATION"), 'yes', 'no')
        self.all_dataset_pd['code'] = 'undetermined'
        self.all_dataset_pd['time'] = 0
        self.total_true = len(self.all_dataset_pd[(self.all_dataset_pd['label'] == 'yes')])
        self.total_false = len(self.all_dataset_pd[(self.all_dataset_pd['label'] == 'no')])

        # logger.info("Ground Truth in total Dataset: True " + str(self.total_true) + " | False " + str(self.total_false))
        return self


    def create_and_process_dataset(self, dataset_names=[], doInclude=True, isTest=False):
        """
        Preprocess by tokenizing and TFIDF vectorizing and creates datasets.
        :param dataset_names: list of dataset names to merge. if given none, all dataset merges into one. will be used
        for cross project validation.
        :param doInclude: should we include the dataset_names or exclude them. Irrelevent if dataset_names is empty
        :return: DATASET class with a csr_mat produced by TFIDF
        """
        if dataset_names:
            if doInclude:
                return DATASET(self.all_dataset_pd.loc[self.all_dataset_pd['projectname'].isin(dataset_names)])
            else:
                return DATASET(self.all_dataset_pd.loc[~self.all_dataset_pd['projectname'].isin(dataset_names)])
        return DATASET(self.all_dataset_pd)


class DATASET:
    def __init__(self, data_pd):
        self.data_pd = data_pd
        self.true_count = len(data_pd[(data_pd['label'] == 'yes')])
        self.false_count = len(data_pd[(data_pd['label'] == 'no')])

        # THIS ONE IS FOR MANUAL TOKENIZATION
        # tfer = TfidfVectorizer(tokenizer=tokenize, preprocessor=None, lowercase=True, stop_words=None, norm='l2',
        #                        use_idf=True, max_features=MAX_FEATURES, decode_error="ignore")

        # logger.info("Ground Truth: True " + str(self.true_count) + " | False " + str(self.false_count))

    def set_csr_mat(self, max_f, stop_w, tfer=None ):
        """

        :param tfer: if training set, give nothing, it will learn and fit_transform. But for Test, it should use
        the tfidf from the training set
        :return:
        """
        if tfer:
            self.tfer = tfer
            self.csr_mat = tfer.transform(self.data_pd['commenttext'])
        else:
            self.tfer = TfidfVectorizer(lowercase=True, stop_words=stop_w, norm='l2',
                                        use_idf=True, max_features=max_f, decode_error="ignore")
            self.csr_mat = self.tfer.fit_transform(self.data_pd['commenttext'])



