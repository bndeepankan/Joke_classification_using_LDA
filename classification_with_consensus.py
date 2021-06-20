"""
@author: bndeepankan
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import pickle


class Consensus:

    def __init__(self, word_category):
        self.word_category = word_category
        df_clean = pd.read_pickle('saved_objects/df_clean_2.pkl')
        with open('saved_objects/doc_top.pkl', 'rb') as f:
            doc_top = pickle.load(f)
        bows = CountVectorizer()
        bows.fit(doc_top)  # learns vocabulary
        self.y = df_clean.label  # label
        self.X = bows.transform(df_clean.jokes_text)  # create document term matrix
        X_features = pd.DataFrame(self.X.toarray(), index=self.y.index,
                             columns=bows.get_feature_names())  # (n_samples X size_feaure_space)
        self.header = bows.get_feature_names()
        pred = self.find_predict(X_features)
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

    def report_classification(self, X, pred):
        """
        return: prints the classification report.
        """
        print('Classification by selecting features. Size of feature space is: {}'.format(X.shape[1]))
        print('==' * 30)
        print('Report of Consensus algorithm')
        print(metrics.classification_report(pred, self.y.to_numpy()))
