#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 11:51:14 2018

@author: arnaud
"""
import numpy as np
import os
import codecs
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

class langageClassifier():
    def __init__(self):
        self.X = []
        self.y = []
        self.loadFiles()
        
    
    def loadFiles(self):
        files = os.listdir('./identification_langue/identification_langue/corpus_entrainement')
        corpus = list(map(lambda f: 
            codecs.open('./identification_langue/identification_langue/corpus_entrainement/'+f, 'r',
                        encoding='latin1').read()
            , files))
        self.files = files
        self.corpus = corpus
        
        for i in range(len(self.files)):
            for sentence in self.tokenize_sentence(self.corpus[i]):
                self.X.append(self.removeUppercase(sentence))
                self.y.append(i)
        
    
    def tokenize_sentence(self, document):
        return(sent_tokenize(document))
        
    def removeUppercase(self, string):
        return string.lower()    
    
    def makeNgram(self, sentence, N = 1):
        idx = np.arange(len(sentence) - N + 1)
        return list(map(lambda i: sentence[i:i+N],idx))

    def fit(self):
        vectorizer = CountVectorizer(tokenizer = word_tokenize, ngram_range = (1,3), analyzer = 'char')
        Xtransform = vectorizer.fit_transform(self.X)
        self.vectorizer = vectorizer
        
        
        cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
        clf = MultinomialNB()
        print('NB score on Cross Val with 10 folds : ' 
              + str(np.mean(cross_val_score(clf, Xtransform, self.y, cv=cv))))  
        clf = LogisticRegression(solver = 'lbfgs', max_iter=100, multi_class = 'auto')
        print('RegLog score on Cross Val with 10 folds : ' 
              + str(np.mean(cross_val_score(clf, Xtransform, self.y, cv=cv))))
        
        clf.fit(Xtransform, self.y)
        self.clf = clf

    def predict(self, doc):
        doc = self.tokenize_sentence(doc)
        doc = list(map(lambda x: self.removeUppercase(x), doc))
        vector = self.vectorizer.transform(doc)
        preds = self.clf.predict(vector)
        pred = 0
        predOccurence = 0
        for i in range(len(self.files)):
            idx = np.where(preds == i)
            if(len(idx[0]) > predOccurence):
                predOccurence = len(idx[0])
                pred = i
        return pred

    def score(self, X_test, y_test):
        pred = self.predict(X_test)
        score = float(len(np.where(pred == y_test))) / float(len(y_test))
        return score 

if __name__ == '__main__':
    clf = langageClassifier()
    clf.fit()
    
    testFiles = os.listdir('./identification_langue/identification_langue/corpus_test1')
    testFiles.sort()
    X_test = list(map(lambda f: 
            codecs.open('./identification_langue/identification_langue/corpus_test1/'+f, 'r',
                        encoding='latin1').read()
            , testFiles))
        
    y_test = [ 'es', 'fr', 'fr', 'en', 'en',
              'fr', 'es', 'es', 'en', 'fr',
              'es', 'fr', 'fr', 'en', 'es',
              'en', 'en', 'fr', 'es', 'fr']
 
    for doc in X_test:
        print(clf.predict(doc))

      