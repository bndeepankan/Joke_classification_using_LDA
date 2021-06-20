"""
@author: bndeepankan
"""

import requests
import json
import pandas as pd
from data_preprocessing import dataPreprocess
from collections import Counter


def verbAtlas(word_list):
    """
    :param word_list: list of words annotated as Noun, verb, etc.
    :return: None
    """

    for tags, word in word_list.items():
        for val in word:
            response = requests.get("http://verbatlas.org/api/verbatlas/predicate?lemma=%s" % val)
            if response.status_code == 200:
                content = json.loads(response.text)
                for id in content.keys():
                    print(val, content[id]['va_frame_id'], content[id]['va_frame_name'])
            else:
                print(val, [])


def topicCategories(doc_top):

    data_for_lda = pd.read_pickle('saved_objects/data_for_lda.pkl').to_dict()['jokes_text']
    word_category = {}
    obj = dataPreprocess()
    categories = obj.categories
    doc_top = set(' '.join(doc_top).split(' '))
    for word in doc_top:
        count_other = 0
        count_related = 0
        for cat, keys in data_for_lda.items():
            try:
                if data_for_lda[cat].index(word):
                    if categories[cat] == 'other':
                        count_other += 1
                    else:
                        count_related += 1
            except ValueError:
                continue
        if count_other > count_related:
            word_category[word] = 'other'
        elif count_other < count_related:
            word_category[word] = 'family_related'
        else:
            word_category[word] = 'common'
    return word_category


def countWords(word_category):

    total = Counter(word_category.values())
    print('total number of words in others: ', total['other'])
    print('total number of words in family related: ', total['family_related'])
    print('total number of words are common for both: ', total['common'])
