#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 22:25:45 2018

@author: arnaud
"""

import nltk
from nltk import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
import numpy as np
import os
import codecs
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn

import warnings
from sklearn.exceptions import ConvergenceWarning


positivesFiles = os.listdir('./books/Book/pos_Bk/')
negativesFiles = os.listdir('./books/Book/neg_Bk/')

class sentimentClassifier():
    #Paramètres par défaut correspondant au meilleur score trouvé pour ces paramètres
    def __init__ (self, params = {
            'lemmatize' : False,
            'stemming' : True,
            'clf': 'log',
            'value': 'presence',
            'openClassOnly' : False,
            'noStopWords': False,
            'min-occur': 3,
            'max-occur': 0.35,
            'addSentiWordNet':True
            }):
        self.params = params
        self.positives, self.negatives = self.loadData() 

    def loadData(self):
        positives = list(map(lambda f: codecs.open('./books/Book/pos_Bk/'+f, 'r', encoding='latin1').read(), positivesFiles))
        negatives = list(map(lambda f: codecs.open('./books/Book/neg_Bk/'+f, 'r', encoding='latin1').read(), negativesFiles))
        return positives, negatives    
    
    def fit(self):
        X = np.append(self.positives, self.negatives)        
        y = np.append(np.ones(len(self.positives)), np.zeros(len(self.negatives)))

        if(self.params['noStopWords']):    
            stop = set(stopwords.words('english'))
            
            #Liste de token ajoutés aux stop words ils correspondent à la liste initiale lemmatizée
            #et stemmisée
            for w in ["'re", "'s", "'ve", 'abov', 'ani', 'becaus', 'becau', 'befor', 'doe', 'dure', 'ha', 'hi', "n't",
                      'need', 'onc', 'onli', 'ourselv', 'sha', 'themselv', 'thi', 'veri', 'wa', 'whi', 'yourselv',
                      "'d", "'ll", 'could', 'might', 'must', 'wo', 'would']:
                stop.add(w)
        else:
            stop = None
        
        #On ne prendra en compte que la présence du mot et pas le nombre de fois où il apparaît dans le document
        if(self.params['value'] == 'count'):
            binary = True
        else:
            binary = False
        
        #Initialisation du vectorizer qui va constituer la matrice d'occurence de toutes les fetures
        #pour tous les documents en entrée
        if(self.params['value'] == 'tf-idf'):
            vectorizerPos = TfidfVectorizer(stop_words = stop, tokenizer = word_tokenize, binary = binary,
                                        preprocessor=self.preprocess, min_df = self.params['min-occur'], max_df = self.params['max-occur'])    
        else:    
            vectorizerPos = CountVectorizer(stop_words = stop, tokenizer = word_tokenize, binary = binary,
                                        preprocessor=self.preprocess, min_df = self.params['min-occur'], max_df = self.params['max-occur'])
        
        #Apprentissage de tous les documents par le vectorizer
        Xtransorm = vectorizerPos.fit_transform(X)

        #On ajoute deux colones à la fin de la matrice qui correspondent pour chaque document
        #au score de positif et de négatif du dit document
        if(self.params['addSentiWordNet']):
            Xtransorm = np.append(np.array(Xtransorm.toarray()), np.array(self.sentiCount(X)), axis = 1)
        
        self.vectorizer = vectorizerPos
        
        #Validation croisée du modèle, on fixe le random_state
        cv = ShuffleSplit(n_splits=3, test_size=0.25, random_state=0)
        if(self.params['clf'] == 'bayesian'):
            clf = MultinomialNB()
            print('NB score on Cross Val with 3 folds : ' 
                  + str(np.mean(cross_val_score(clf, Xtransorm, y, cv=cv))))  
        else:
            clf = LogisticRegression(solver = 'lbfgs')
            print('RegLog score on Cross Val with 3 folds : ' 
                  + str(np.mean(cross_val_score(clf, Xtransorm, y, cv=cv))))
        
        
        #Apprentissage final sur tout le jeu de donnée
        clf.fit(Xtransorm, y)
        self.clf = clf

        
    #Pour chaque mot du document, on applique un stemming et on reconstitue un
    #document avec toutes les racines séparées par des espaces
    def stem(self, string):
        porter_stemmer = nltk.stem.porter.PorterStemmer()
        tokenizedStr = word_tokenize(string)
        stemsStr = list(map(lambda word: porter_stemmer.stem(word), tokenizedStr))
        space = " "
        return space.join(stemsStr)
    
    #Pour chaque mot du document, on applique une lemmatization et on reconstitue un
    #document avec touts les lemmes séparés par des espaces
    def lemm(self, string):    
        wordnet_lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
        tokenizedStr = word_tokenize(string)
        lemsStr = list(map(lambda word: wordnet_lemmatizer.lemmatize(word), tokenizedStr))
        space = " "
        return space.join(lemsStr)
 
        return lemsStr

    #Soit la liste des tags correspondant aux classes ouvertes
    #On ne garde que les mots tagés par ces classes
    def removeClosedClasses(self, string):   
        open_classes = {'FW', 'JJ', 'JRJ', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS',
                        'RB', 'RBR', 'RBS', 'UH', 'VB', 'VBD', 'VBG', 'VBN',
                        'VBP', 'VBZ'}    
        wordArray = word_tokenize(string)
        taggedWords = pos_tag(wordArray)
        noClosedClass = list(filter(lambda tagged_word: tagged_word[1] in open_classes,taggedWords))
        noClosedClass = list(map(lambda x: x[0], noClosedClass))
        space = " "
        return space.join(noClosedClass)
    
    #Le preprocess est un argument du vectorizer (qui extrait les features)
    #Cette fonction reprend les fonctions ci dessus et sera appliquée à chaque document
    def preprocess(self, string):
        if(self.params['openClassOnly']):
            string = self.removeClosedClasses(string)
        
        if(self.params['lemmatize']):
            string = self.lemm(string)
        elif(self.params['stemming']):
            string = self.stem(string)

        #print(string)
        return string
    
    #Cette fonction simplifie les tag de nltk en tag comprehensible par sentiwordnet
    def penn_to_wn(self, tag):
        """
        Convert between the PennTreebank tags to simple Wordnet tags
        """
        if tag.startswith('J'):
            return wn.ADJ
        elif tag.startswith('N'):
            return wn.NOUN
        elif tag.startswith('R'):
            return wn.ADV
        elif tag.startswith('V'):
            return wn.VERB
        return None
    
    def sentiCount(self, X):
        sentiScore = np.zeros((len(X),2))
        #Pour chaque document on calcule le score de positivité et de négativité associé
        for i in range(len(X)):
            
            posScore = 0.0
            negScore = 0.0
            tokenCount = 0.0
            wordArray = word_tokenize(X[i])
            taggedWords = pos_tag(wordArray)
            
            #Pour chaque mot, sentiwordnet renvoie un score de positivité négativité
            #selon le pos tag de ce mot 
            for word,tag in taggedWords:
                tag = self.penn_to_wn(tag)
                if tag not in (wn.NOUN, wn.ADJ, wn.ADV, wn.VERB):
                    continue
                synsets = wn.synsets(word, pos=tag)
                if not synsets:
                    continue
                
                synset = synsets[0]
                swn_synset = swn.senti_synset(synset.name())
                posScore += swn_synset.pos_score()
                negScore += swn_synset.neg_score()
                tokenCount += 1.0
            
            #Afin d'avoir des valeurs du même ordre de grandeur dans la matrice final
            #qui entrainera le classifieur, on peut choisir de rammener entre 0 ou 1 
            #la somme du score de tous les mots (somme vs moyenne)
            if(self.params['value'] == 'count'):
                sentiScore[i,0] = posScore
                sentiScore[i,1] = negScore
            else:
                if(tokenCount != 0.0):
                    sentiScore[i,0] = posScore / tokenCount
                    sentiScore[i,1] = negScore / tokenCount
                else:
                    sentiScore[i,0] = 0.0
                    sentiScore[i,1] = 0.0
            
        return sentiScore
            
    #On applique les mêmes prétraitements que précédemment
    def predict(self, string):
        vector = self.vectorizer.transform(string)
        
        if(self.params['addSentiWordNet']):
            vector = np.append(np.array(vector.toarray()), np.array(self.sentiCount(string)), axis = 1)
        
        print(self.clf.predict(vector))
        
    
if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    clf = sentimentClassifier(params = {
            'lemmatize' : False,
            'stemming' : True,
            #clf peut avoir la valeur 'log' ou 'bayesian'
            'clf': 'log',
            #'presence', 'count', 'tf-idf'
            'value': 'count',
            'openClassOnly' : True,
            'noStopWords': False,
            'min-occur': 1,
            'max-occur': 1.0,
            'addSentiWordNet':True
            })
    clf.fit()
    #clf.predict(["This movie was the most awfull movie I ever saw ... Really trash I don't recommand it You will waste yor monney it'a a scam"])
    
