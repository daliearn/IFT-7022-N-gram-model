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
    def __init__(self, N = 3):
        self.X = []
        self.y = []
        self.loadFiles()
        self.N = N
    
    def loadFiles(self):
        files = os.listdir('./identification_langue/identification_langue/corpus_entrainement')
        files.sort()
        corpus = list(map(lambda f: 
            codecs.open('./identification_langue/identification_langue/corpus_entrainement/'+f, 'r',
                        encoding='latin1').read()
            , files))
        self.files = files
        self.corpus = corpus
        
        #On passe tous les documents en lowercase
        #On les segmente également en phrases
        for i in range(len(self.files)):
            for sentence in self.tokenize_sentence(self.corpus[i]):
                self.X.append(self.removeUppercase(sentence))
                self.y.append(i)
        
    
    def tokenize_sentence(self, document):
        return(sent_tokenize(document))
        
    def removeUppercase(self, string):
        return string.lower()  
    
    #Cette fonction été demandée mais n'est pas nécessaire
    #Count vectorizer s'en charge
    def makeNgram(self, sentence, N = 1):
        idx = np.arange(len(sentence) - N + 1)
        return list(map(lambda i: sentence[i:i+N],idx))

    def fit(self):
        #Count Vectorizer will annalyze character by character all documents to construct the model
        vectorizer = CountVectorizer(tokenizer = word_tokenize, ngram_range = (1,self.N), analyzer = 'char')
        Xtransform = vectorizer.fit_transform(self.X)
        self.vectorizer = vectorizer
        
        
        #Evaluation using cross validation
        cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
        clf = MultinomialNB()
        print('NB score on Cross Val with 10 folds : ' 
              + str(np.mean(cross_val_score(clf, Xtransform, self.y, cv=cv))))  
        clf = LogisticRegression(solver = 'lbfgs', max_iter=100, multi_class = 'auto')
        print('RegLog score on Cross Val with 10 folds : ' 
              + str(np.mean(cross_val_score(clf, Xtransform, self.y, cv=cv))))
        
        #Entrainement sur le jeu de test complet
        clf = MultinomialNB()
        clf.fit(Xtransform, self.y)
        self.clf = clf

    #0 : english
    #1 : espanol
    #2 : français
    #3 : portuguese
    def predict(self, doc):
        #Preprocessing to remove upercases
        doc = self.tokenize_sentence(doc)
        doc = list(map(lambda x: self.removeUppercase(x), doc))
        
        #document vectorization 
        vector = self.vectorizer.transform(doc)
        
        #Le classifieur va retourner une liste de predictions corresponndants
        #a une prediction pour chaque phrase du document
        preds = self.clf.predict(vector)
        
        #Puisqu'oon a plusieurs predictions on va procéder à un vote majoritaire  
        #L'index avec le plus d'apparition sera la prédiction
        pred = 0
        predOccurence = 0
        for i in range(len(self.files)):
            idx = np.where(preds == i)
            if(len(idx[0]) > predOccurence):
                predOccurence = len(idx[0])
                pred = i
        return pred


    def score(self, X_test, y_test):
        preds = np.zeros(len(y_test))
        for i in range(len(X_test)):
            preds[i] = int(self.predict(X_test[i]))
        
        #On fait juste le compte de predictions que l'on divise par le nombre de predictions
        score = float(len(np.where(preds == y_test)[0])) / float(len(y_test))
        return score 

if __name__ == '__main__':
    clf = langageClassifier(3)
    clf.fit()
    
    testFiles = os.listdir('./identification_langue/identification_langue/corpus_test1')
    testFiles.sort()
    
    
    X_test = list(map(lambda f: 
            codecs.open('./identification_langue/identification_langue/corpus_test1/'+f, 'r',
                        encoding='latin1').read()
            , testFiles))
    
    #On peut aller voir dans clf.files à quoi correspondent ces indices 
    #0 : english
    #1 : espanol
    #2 : français
    #3 : portuguese
    #print(clf.files)
    y_test = [ 1, 2, 1, 2, 2,
              0, 1, 0, 0, 2,
              1, 2, 2, 3, 3,
              3, 3, 3, 3, 3,
              3, 3, 2, 3, 2, 
              2, 0, 0, 0, 0, 
              1, 1, 1, 0, 1, 
              0, 2, 1, 1, 0]
    print('score sur les fichiers de test')
    print(clf.score(X_test, y_test))
    
    print("pour utiliser la prediction, faites clf.predict(string)")
    print("pour interpreter la prediction, faites print(clf.files[pred])")   
    
    