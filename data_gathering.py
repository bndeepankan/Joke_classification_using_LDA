# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 17:56:21 2021

@author: Alibay
"""
import requests
from bs4 import BeautifulSoup
import time


jokes_pages={'animal-jokes': 24,
              'blonde-jokes': 18,
              'boycott-these-jokes': 5,
              'clean-jokes': 35,
              'family-jokes': 13,
              'food-jokes': 7,
              'holiday-jokes': 4,
              'insult-jokes': 22, 
              'national-jokes': 4,
              'office-jokes': 5,
              'political-jokes': 4,
              'relationship-jokes': 14,
              'religious-jokes': 9,
              'school-jokes': 7,
              'science-jokes': 3,
              'sex-jokes': 23,
              'sexist-jokes': 7,
              'sports-jokes': 4,
              'technology-jokes': 3}
#running this file will take around 30 minutes. Data in jokes folder
for jokes in jokes_pages:
    url = "http://www.laughfactory.com/jokes/"
    url = url + jokes + "/"
    count=1
    for i in range(1,jokes_pages[jokes]+1): # extract joke from multiple pages
        r = requests.get(url + str(i))
        soup = BeautifulSoup(r.text, 'lxml')
    
        for joke in soup.find_all('div',class_='joke-text'):
            with open('jokes/'+jokes+'.txt','a',encoding='utf-8') as f:
                f.write(str(count) + '. ' + joke.p.text.strip()+'\n')
            count+=1
        time.sleep(7) #to prevent overloading of web server

