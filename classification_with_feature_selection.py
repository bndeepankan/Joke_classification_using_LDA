# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 22:52:24 2021

@author: Alibay
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from classification_with_full_features import X, y, X_train, X_test, y_train, y_test
import pickle


def prepare_data_for_sklearn(df_clean, doc_top):
    bows = CountVectorizer()
    bows.fit(doc_top)  # learns vocabulary
    X_few = bows.transform(df_clean.jokes_text) # create document term matrix  
    X_few = pd.DataFrame(X_few.toarray(), index=y.index, columns=bows.get_feature_names()) #(n_samples X size_feaure_space)
    
    # choose the same observations which is used in full features classification
    X_train_few = X_few.loc[X_train.index, :]
    X_test_few = X_few.loc[X_test.index, :]
    y_train_few = y_train
    y_test_few = y_test
    return X_train_few, X_test_few, y_train_few, y_test_few, X_few


def train_and_predict_data(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train) # train model
    y_pred = model.predict(X_test)
    return y_pred


def report_classification(X_few, y_test, y_pred, model, alpha=None):
    if isinstance(model, LogisticRegression):
        print('Classification by selecting features. Size of feature space is: {}'.format(X_few.shape[1]) )
        print('=='*30)
        print('Report of Logistic Regression')
        print(metrics.classification_report(y_test, y_pred))
    else:
        print('=='*30)
        print('Report of Multinomial Naive Bayes with best the hyperparameter alpha={:.1f}'.format(alpha))
        print(metrics.classification_report(y_test, y_pred))
        print('=='*30)


def start():
    df_clean = pd.read_pickle('saved_objects/df_clean_2.pkl')
    with open('saved_objects/doc_top.pkl', 'rb') as f:
        doc_top = pickle.load(f)
    X_train, X_test, y_train, y_test, X_few = prepare_data_for_sklearn(df_clean, doc_top)
    
    #Logistic Regression trainingl
    logreg = LogisticRegression()
    y_pred = train_and_predict_data(X_train, X_test, y_train, y_test, logreg)
    report_classification(X_few, y_test, y_pred, logreg)
    
    
    #Mulinomial Naive Bayes training
    mnb = MultinomialNB()
    param_grid = {'alpha':[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]}
    mnb = MultinomialNB()
    grid = GridSearchCV(mnb, param_grid, scoring='accuracy', cv=10) #use grid search to find best hyperparameter
    grid.fit(X_few, y)
    
    mnb = MultinomialNB(alpha=grid.best_params_['alpha'])
    y_pred = train_and_predict_data(X_train, X_test, y_train, y_test, mnb)
    report_classification(X_few, y_test, y_pred, mnb, grid.best_params_['alpha'])




