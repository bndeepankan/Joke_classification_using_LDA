# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 17:55:05 2021

@author: Alibay
"""
import sys
from LDA_implementation import LDA
import classification_with_full_features as cwfl
import classification_with_feature_selection as cwfs
import time
from annotate_data import verbAtlas
from annotate_data import topicCategories
from annotate_data import countWords
from classification_with_consensus import Consensus

if __name__ == '__main__':
    number_topics, number_words= int(sys.argv[1]), int(sys.argv[2])
    if len(sys.argv) == 4:
        seed = int(sys.argv[3])
    # number_topics, number_words, seed = 14, 40, 5
    else:
        seed = None
    print('LDA generates topics...')
    obj_lda = LDA(number_topics, number_words, seed)
    word_list = obj_lda.word_list
    # print('Finding the top frames with VerbAtlas')
    # verbAtlas(word_list)
    word_category = topicCategories(obj_lda.doc_top)
    # print(countWords(word_category))
    obj_con = Consensus(word_category)
    start_cwfl = time.time()
    cwfl.start()
    end_cwfl = time.time()
    print()
    print('\n')
    start_cwfs = time.time()
    cwfs.start()
    end_cwfs = time.time()
    print('\n')
    print('Runtime of Classification with full features: {:.2f}'.format(end_cwfl - start_cwfl))
    print('Runtime of Classification with feature selection: {:.2f}'.format(end_cwfs - start_cwfs))
