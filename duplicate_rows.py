# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 18:30:34 2021

@author: Alibay
"""
import pandas as pd

def remove_duplicates(df_clean):
    df_clean.drop_duplicates(keep='first',inplace=True) # in each duplicates keep only the 1st one and delete others
    df_duplicates = df_clean[df_clean['jokes_text'].duplicated(keep=False)] #duplicates, such that has the same joke text but has different label
    ser = (df_duplicates.label == 'family_related') #delete all familty_related duplicates
    df_clean.drop(labels=ser[ser].index, axis=0, inplace=True)
    index = [0, 3, 16, 21, 52, 54, 58, 60, 61, 62, 103, 113, 135, 151, 169, 196, 203, 
        215, 1196, 1865, 1878, 2377, 76, 77] #delete those indeces from 'other' class. These are duplicates
    df_clean.drop(labels=index, axis=0, inplace=True)
    
    return df_clean

df_clean = pd.read_pickle('saved_objects\\df_clean.pkl')
df_clean = remove_duplicates(df_clean)
df_clean.to_pickle('saved_objects\\df_clean_2.pkl')