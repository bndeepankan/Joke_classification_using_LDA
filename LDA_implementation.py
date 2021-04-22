# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 19:38:08 2021

@author: Alibay
"""
from gensim import models
from gensim.corpora import Dictionary
import re
import pandas as pd
import pickle
import numpy as np


def train_lda(data_for_lda, number_topics, seed=None):
    docs = list(data_for_lda.jokes_text) # create list of list
    dct = Dictionary(docs)
    corpus = [dct.doc2bow(doc) for doc in docs] # create BOW model for each document
    np.random.seed(seed)
    lda = models.LdaModel(corpus, num_topics=number_topics, id2word=dct, passes=10)
    return lda

def extract_top_words(lda, number_words):
    from data_preprocessing import tokens_to_sents
    topics = lda.print_topics(num_words= number_words)
    doc_top = []
    for topic in topics:
        s = topic[1]
        tokens = re.findall(r'[a-z]+', s) #extract only words from string
        doc_top.append(tokens_to_sents(tokens))
    return doc_top

def run_lda(number_topics, number_words, seed):
    data_for_lda = pd.read_pickle('saved_objects\\data_for_lda.pkl')
    lda = train_lda(data_for_lda, number_topics, seed)
    doc_top = extract_top_words(lda, number_words)
    with open('saved_objects\\doc_top.pkl', 'wb') as f:
        pickle.dump(doc_top, f)
    print('Topics generation finished and saved in \\saved_objects\\doc_top.pkl')
    #print(doc_top)







    