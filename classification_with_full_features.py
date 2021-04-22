# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 20:08:25 2021

@author: Alibay
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

def prepare_data_for_sklearn(df_clean):
    bows = CountVectorizer()
    X = bows.fit_transform(df_clean.jokes_text) # create document term matrix  
    X = pd.DataFrame(X.toarray(), index=df_clean.index, columns=bows.get_feature_names()) #(n_samples X size_feaure_space)
    
    return X

def train_and_predict_data(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train) # train model
    y_pred = model.predict(X_test)
    return y_pred

def report_classification(X, y_test, y_pred, model, alpha=None):
    if isinstance(model, LogisticRegression):
        print('\n')
        print('Classification using full features. Size of feature space is: {}'.format(X.shape[1]))
        print('=='*30)
        print('Report of Logistic Regression using Full Features')
        print(metrics.classification_report(y_test, y_pred))
    else:
        print('=='*30)
        print('Report of Multinomial Naive Bayes using Full Features with the best hyperparameter alpha={:.1f}'.format(alpha))
        print(metrics.classification_report(y_test, y_pred))
        print('=='*30)


def start():
    
    #Logistic Regression trainingl
    logreg = LogisticRegression()
    y_pred = train_and_predict_data(X_train, X_test, y_train, y_test, logreg)
    report_classification(X, y_test, y_pred, logreg)
    
    
    #Mulinomial Naive Bayes training
    mnb = MultinomialNB()
    param_grid = {'alpha':[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]}
    mnb = MultinomialNB()
    grid = GridSearchCV(mnb, param_grid, scoring='accuracy', cv=10) #use grid search to find best hyperparameter
    grid.fit(X, y)
    
    mnb = MultinomialNB(alpha=grid.best_params_['alpha'])
    y_pred = train_and_predict_data(X_train, X_test, y_train, y_test, mnb)
    report_classification(X, y_test, y_pred, mnb, grid.best_params_['alpha'])


if __name__ == 'classification_with_full_features':
    df_clean = pd.read_pickle('saved_objects\\df_clean_2.pkl')
    X = prepare_data_for_sklearn(df_clean)
    y = df_clean.label #label 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3) # 30 percent for testing

    