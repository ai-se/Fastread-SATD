import string

import pandas
import numpy as np
from nltk import WordNetLemmatizer, sent_tokenize, wordpunct_tokenize, pos_tag


# CITE TODO RAHUL
def tokenize(document):
    lemmatizer = WordNetLemmatizer()

    "Break the document into sentences"
    for sent in sent_tokenize(document):

        "Break the sentence into part of speech tagged tokens"
        for token, tag in pos_tag(wordpunct_tokenize(sent)):

            "Apply preprocessing to the token"
            token = token.lower()  # Convert to lower case
            token = token.strip()  # Strip whitespace and other punctuations
            token = token.strip('_')  # remove _ if any
            token = token.strip('*')  # remove * if any

            "If stopword, ignore."
            # if token in stopwords.words('english'):
            #     continue

            "If punctuation, ignore."
            if all(char in string.punctuation for char in token):
                continue

            "If number, ignore."
            if token.isdigit():
                continue

            # Lemmatize the token and yield
            # Note: Lemmatization is the process of looking up a single word form
            # from the variety of morphologic affixes that can be applied to
            # indicate tense, plurality, gender, etc.
            lemma = lemmatizer.lemmatize(token)
            yield lemma


# CITE TODO https://buhrmann.github.io/tfidf-analysis.html
# Also have a plot function and a few more
def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pandas.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

def top_feats_in_doc(Xtr, features, row_id, top_n=25):
    ''' Top tfidf features in specific document (matrix row) '''
    row = np.squeeze(Xtr[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)

def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    ''' Return the top n features that on average are most important amongst documents in rows
        indentified by indices in grp_ids. '''
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)

def top_feats_by_class(Xtr, y, features, min_tfidf=0.1, top_n=25):
    ''' Return a list of dfs, where each df holds top_n features and their mean tfidf value
        calculated across documents with the same class label. '''
    dfs = []
    labels = np.unique(y)
    for label in labels:
        ids = np.where(y==label)
        feats_df = top_mean_feats(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs

def process_output(satdd, filename, append_str, rig_type='cross'):
    datasets = satdd.all_dataset_pd.projectname.unique()
    output = ''
    with open("logs/" + filename, "r") as f:
        for line in f.readlines():
            if "Optimized" in line:
               output += line + '\n'

    file = open('results/' + append_str + rig_type +'_proj_f_only.txt', 'w')
    file.write(str(output))

    avg_fs = ''
    for dataset in datasets:
        output = ''
        avg_f = 0
        with open("logs/" + filename, "r") as f:
            for line in f.readlines():
                if dataset in line:
                    output += line + '\n'
                    if "Optimized F Score for fold" in line:
                        avg_f += float(line.split(":")[3])

        avg_f = avg_f/5
        avg_fs += dataset + " " + str(avg_f) + "\n"

        file = open('results/' + append_str + rig_type + '_proj_' + dataset + ".txt", 'w')
        file.write(str(output))

    file = open('results/' + append_str + rig_type + '_proj_avg_f.txt', 'w')
    file.write(str(avg_fs))


def compare_with_huang(filename, write_filename):
    res_dict = {}
    with open("results/huang_cross_results.txt", "r") as huang:
        for line in huang.readlines():
            temp = line.split(" ")
            res_dict[temp[0]] = temp[1]

    output = 'Project,F,Difference\n'
    with open(filename, "r") as file:
        for line in file.readlines():
            temp = line.split(" ")
            val = res_dict.get(temp[0])
            val= round((float(temp[1]) - float(val)),3)
            output += temp[0] + "," + str(round(float(temp[1]) * 100, 2)) + "," + str(round(val * 100, 2)) + "\n"

    f = open(write_filename, "w+")
    f.write(output)