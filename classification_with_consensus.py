"""
@author: bndeepankan
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import pickle


class Consensus:

    def __init__(self, word_category, pattern_list):
        self.word_category = word_category
        self.pattern_list = pattern_list
        self.df_clean = pd.read_pickle('saved_objects/df_test.pkl')
        with open('saved_objects/doc_top.pkl', 'rb') as f:
            doc_top = pickle.load(f)
        bows = CountVectorizer()
        bows.fit(doc_top)  # learns vocabulary
        self.y = self.df_clean.label  # label
        self.X = bows.transform(self.df_clean.jokes_text)  # create document term matrix
        X_features = pd.DataFrame(self.X.toarray(), index=self.y.index,
                             columns=bows.get_feature_names())  # (n_samples X size_feaure_space)
        self.header = bows.get_feature_names()
        pred = self.find_prob_predict(X_features)
        self.report_classification(X_features, pred)
        pred = self.find_match_predict()
        self.report_classification(X_features, pred)

    def find_predict(self, X):
        """
        return: prediction of values on the input training set.
        """
        count = {'other': 0, 'family_related': 0, 'common': 0}
        pred = []
        tmp = 0
        for ind in X.index:
            for topic in self.header:
                if X[topic][ind] > 0:
                    count[self.word_category[topic]] += 1
            val = max(count, key=count.get)
            if val == 'common':
                tmp += 1
            pred.append(max(count, key=count.get))
        print('common types occur: ', tmp)
        return pred

    def find_prob_predict(self, X):
        """
        prediction of values on probability of input training set
        """
        count = {'other': 0, 'family_related': 0}
        pred = []
        for ind in X.index:
            for topic in self.header:
                if X[topic][ind] > 0:
                    count['family_related'] += self.word_category[topic][0]
                    count['other'] += self.word_category[topic][1]
            if count['other'] == count['family_related']:
                print(ind)
            pred.append(max(count, key=count.get))
        return pred

    def find_match_predict(self):
        """
        Match with the pattern in the training.
        return: prediction values
        """
        jokes_text, labels = self.df_clean['jokes_text'].tolist(), self.df_clean['label'].tolist()
        result = []
        for joke in range(len(jokes_text)):
            pattern = []
            for token in jokes_text[joke].split(' '):
                try:
                    if self.word_category[token] and len(set(self.word_category[token])) > 1:
                        ind = self.word_category[token].index(max(self.word_category[token]))
                        pattern.append('FR' if not ind else 'OT')
                    else:
                        pattern.append('E')
                except:
                    continue
            try:
                result.append(self.pattern_list[tuple(pattern)])
            except:
                result.append('family_related')
        return result

    def report_classification(self, X, pred):
        """
        return: prints the classification report.
        """
        target = self.y.to_numpy()
        print('Classification by selecting features. Size of feature space is: {}'.format(X.shape[1]))
        print('==' * 30)
        print('Report of Consensus algorithm')
        print(metrics.classification_report(pred, target))
