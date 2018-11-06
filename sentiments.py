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

positivesFiles = os.listdir('./books/Book/pos_Bk/')
negativesFiles = os.listdir('./books/Book/neg_Bk/')

class sentimentClassifier():
    def __init__ (self, params = {
            'lemmatize' : True,
            'stemming' : False,
            'openClassOnly' : True,
            'noStopWords': True,
            'clf': 'log',
            'value': 'frequency',
            'min-occur': 0.0,
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
            
            #List of stop word tokenized "should be better
            for w in ["'re", "'s", "'ve", 'abov', 'ani', 'becaus', 'becau', 'befor', 'doe', 'dure', 'ha', 'hi', "n't",
                      'need', 'onc', 'onli', 'ourselv', 'sha', 'themselv', 'thi', 'veri', 'wa', 'whi', 'yourselv']:
                stop.add(w)
        else:
            stop = None
            
        if(self.params['value'] == 'frequency'):
            binary = True
        else:
            binary = False
            
        if(self.params['value'] == 'tf-idf'):
            vectorizerPos = TfidfVectorizer(stop_words = stop, tokenizer = word_tokenize, binary = binary,
                                        preprocessor=self.preprocess, min_df = self.params['min-occur'])    
        else:    
            vectorizerPos = CountVectorizer(stop_words = stop, tokenizer = word_tokenize, binary = binary,
                                        preprocessor=self.preprocess, min_df = self.params['min-occur'])

        Xtransorm = vectorizerPos.fit_transform(X)

        if(self.params['addSentiWordNet']):
            Xtransorm = np.append(np.array(Xtransorm.toarray()), np.array(self.sentiCount(X)), axis = 1)
        
        self.vectorizer = vectorizerPos
        
        #TODO : tester une implémentation soit meme avec repeated k fold
        cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
        if(self.params['clf'] == 'bayesian'):
            clf = MultinomialNB()
            print('NB score on 10 Cross Val with 10 folds : ' 
                  + str(np.mean(cross_val_score(clf, Xtransorm, y, cv=cv))))  
        else:
            clf = LogisticRegression(solver = 'lbfgs')
            print('RegLog score on 10 Cross Val with 10 folds : ' 
                  + str(np.mean(cross_val_score(clf, Xtransorm, y, cv=cv))))
        
        clf.fit(Xtransorm, y)
        self.clf = clf

        
    
    def stem(self, string):
        porter_stemmer = nltk.stem.porter.PorterStemmer()
        #Tokenization
        tokenizedStr = word_tokenize(string)
        stemsStr = list(map(lambda word: porter_stemmer.stem(word), tokenizedStr))
        space = " "
        return space.join(stemsStr)
    
    def lemm(self, string):    
        wordnet_lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
        tokenizedStr = word_tokenize(string)
        lemsStr = list(map(lambda word: wordnet_lemmatizer.lemmatize(word), tokenizedStr))
        space = " "
        return space.join(lemsStr)
 
        return lemsStr

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
    
    
    def preprocess(self, string):
        if(self.params['lemmatize'] == True):
            string = self.lemm(string)
        elif(self.params['stemming'] == True):
            string = self.stem(string)

        if(self.params['openClassOnly'] == True):
            string = self.removeClosedClasses(string)
        return string
    
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
        for i in range(len(X)):
            posScore = 0.0
            negScore = 0.0
            tokenCount = 0.0
            wordArray = word_tokenize(X[i])
            taggedWords = pos_tag(wordArray)
            for word,tag in taggedWords:
                tag = self.penn_to_wn(tag)
                if tag not in (wn.NOUN, wn.ADJ, wn.ADV):
                    continue
                synsets = wn.synsets(word, pos=tag)
                if not synsets:
                    continue
                
                synset = synsets[0]
                swn_synset = swn.senti_synset(synset.name())
                posScore += swn_synset.pos_score()
                negScore += swn_synset.neg_score()
                tokenCount += 1.0
            
            if(tokenCount != 0.0):
                sentiScore[i,0] = posScore / tokenCount
                sentiScore[i,1] = negScore / tokenCount
            else:
                sentiScore[i,0] = 0.0
                sentiScore[i,1] = 0.0
        return sentiScore
            
    def predict(self, string):
        vector = self.vectorizer.transform(string)
        
        if(self.params['addSentiWordNet']):
            vector = np.append(np.array(vector.toarray()), np.array(self.sentiCount(string)), axis = 1)
        
        print(self.clf.predict(vector))
        
    
if __name__ == '__main__':
    clf = sentimentClassifier()
    clf.fit()
    clf.predict(["This movie was the most awfull shit I ever saw ..."])


