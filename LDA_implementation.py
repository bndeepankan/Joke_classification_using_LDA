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
from data_preprocessing import dataPreprocess


class LDA():

    def __init__(self, number_topics, number_words, seed):
        self.obj = dataPreprocess()
        data_for_lda = pd.read_pickle('saved_objects/data_for_lda.pkl')
        lda = self.train_lda(data_for_lda, number_topics, seed)
        self.doc_top = self.extract_top_words(lda, number_words)
        with open('saved_objects/doc_top.pkl', 'wb') as f:
            pickle.dump(self.doc_top, f)
        print('Topics generation finished and saved in /saved_objects/doc_top.pkl')

        self.word_list = self.print_keyword(self.doc_top)

    def train_lda(self, data_for_lda, number_topics, seed=None):
        docs = list(data_for_lda.jokes_text) # create list of list
        dct = Dictionary(docs)
        corpus = [dct.doc2bow(doc) for doc in docs] # create BOW model for each document
        np.random.seed(seed)
        lda = models.LdaModel(corpus, num_topics=number_topics, id2word=dct, passes=10)
        return lda

    def extract_top_words(self, lda, number_words):
        topics = lda.print_topics(num_words= number_words)
        doc_top = []
        for topic in topics:
            s = topic[1]
            tokens = re.findall(r'[a-z]+', s) #extract only words from string
            doc_top.append(self.obj.tokens_to_sents(tokens))
        return doc_top

    def print_keyword(self, doc_top):
        """
        :param doc_top: contains keywords based on LDA
        :return: keywords based on nouns, verbs etc.
        """
        wordList = {}
        for tp in range(len(doc_top)):
            text = self.obj.tag_text(doc_top[tp])
            for keyword, tag in text:
                try:
                    wordList[tag].add(keyword)
                except:
                    wordList[tag] = {keyword}
        # print(wordList)
        return wordList
