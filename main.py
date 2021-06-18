# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 17:55:05 2021

@author: Alibay
"""
import sys
import LDA_implementation
import classification_with_full_features as cwfl
import classification_with_feature_selection as cwfs
import time
from annotate_data import verbAtlas

if __name__ == '__main__':
    number_topics, number_words= int(sys.argv[1]), int(sys.argv[2])
    if len(sys.argv) == 4:
        seed = int(sys.argv[3])
    #number_topics, number_words, seed = 14, 40, 5
    else:
        seed = None
    print('LDA generates topics...')
    word_list = LDA_implementation.run_lda(number_topics, number_words, seed)
    # print('Finding the top frames with VerbAtlas')
    verbAtlas(word_list)
    """
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
    """
