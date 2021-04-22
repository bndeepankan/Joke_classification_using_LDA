# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 18:03:53 2021

@author: Alibay
"""
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import pandas as pd
import re
import os

jokes_list=['animal', 'blonde', 'boycott', 'clean', 'family', 'food', 'holiday','insult', 'national', 'office', 
      'political', 'relationship', 'religious', 'school', 'science', 'sex', 'sexist', 'sports', 'technology']

#read jokes and store them in a dataframe.In a document each jokes starts with a number
def read_jokes(jokes_list):
    for joke in jokes_list:
        with open('jokes/'+ joke +'-jokes.txt', encoding='utf-8') as f: #jokes folder where all joke documents are there
            jokes = [] 
            new_joke=''
            for line in f:
                found = re.findall(r'^(\d+\.)', line) # if line starts with digits then it is a new joke
                if len(found) > 0: #new joke is found, add previous joke to the list
                    jokes.append(new_joke)
                    new_joke = '' 
                    new_joke = new_joke + line
                else:
                    new_joke = new_joke + line
            jokes.remove('')
            if joke == 'animal': #if it is first file just create Dataframe. 'animal-jokes.txt' is the first file
                df_raw = pd.DataFrame(jokes)
                df_raw['label'] = joke
            else:                # if it is not first file then concat dataframes
                df_temp = pd.DataFrame(jokes)
                df_temp['label'] = joke
                df_raw = pd.concat([df_raw, df_temp], ignore_index=True)
    return df_raw

def get_pos(token):
    # get pos of a token. It will be used in lemmatization process
    pos = pos_tag([token])[0][1][0]
    if pos == 'V':
        return wordnet.VERB
    elif pos == 'J':
        return wordnet.ADJ
    elif pos == 'R':
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess(text):
    stop_words = stopwords.words('english')
    new_stop_words = ['say', 'ask', 'tell', 'get','come', 'see', 'asks', 'go','know', 'look', 'first','second', 
                  'walk','see','one', 'two', 'reply', 'like', 'make', 'back','would','man']
    stop_words = stop_words + new_stop_words 
    wnl = WordNetLemmatizer()
    text = text.lower() #make lowercase
    tokens = word_tokenize(text) 
    tokens = [token for token in tokens if token.isalpha()] #if a token is not a digit or not contains any digits then add
    tokens = [wnl.lemmatize(token, get_pos(token)) for token in tokens] # lemmatize a token taken into account its pos tag
    tokens = [token for token in tokens if token not in stop_words] #remove stopwords
    tokens = [token for token in tokens if len(token)>2]
    return tokens

#convert list of tokens to the string
def tokens_to_sents(tokens): 
    return ' '.join(tokens)

def noun_adj(text):
    # Keep only nouns and adjectives
    text = [token for (token, pos) in pos_tag(text) if ( pos[:2]=='NN' or pos[:2] == 'JJ')]
    return text 

df_raw = read_jokes(jokes_list)
df_raw.rename({0: 'jokes_text'}, axis=1, inplace=True) #change column names to 'jokes_text'
df_raw['label'] = df_raw.label.map({'animal':'other', 'blonde': 'family_related', 'boycott': 'family_related', 
                 'clean':'family_related', 'family': 'family_related','food': 'family_related', 'holiday': 'other', 
                 'insult': 'family_related', 'national':'other', 'office': 'family_related', 
                 'political':'other', 'relationship': 'family_related',  'religious': 'other', 
                 'school': 'family_related', 'science': 'other', 'sex': 'family_related',
                 'sexist': 'family_related', 'sports': 'other', 'technology': 'other'}) 
df_raw_prep = df_raw['jokes_text'].apply(preprocess)
df_clean = df_raw_prep.apply(tokens_to_sents) #apply tokens_to_sents function to series values
df_clean = pd.DataFrame(df_clean) #convert to df

df_clean['label'] = df_raw['label'] # add 'label' column 

# read each file to the dataframe such that row of a dataframe corresponds to one document
data = {} 
for i, f in enumerate(os.listdir('jokes\\')):
    with open('jokes\\'+f, mode ='r', encoding='utf-8') as f_obj:
        data[jokes_list[i]] = [f_obj.read()]

data_for_lda = pd.DataFrame(data).T
data_for_lda.columns = ["jokes_text"]
data_for_lda = data_for_lda.jokes_text.apply(preprocess)
data_for_lda = pd.DataFrame(data_for_lda.apply(noun_adj)) #keep only nouns and adjectives
data_for_lda = pd.DataFrame(data_for_lda)

#Save the dataframes:
df_clean.to_pickle('saved_objects\\df_clean.pkl') # it will be used in classification
data_for_lda.to_pickle('saved_objects\\data_for_lda.pkl') # it will be used for topic extraction
