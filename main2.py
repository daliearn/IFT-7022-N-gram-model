#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
#nltk.download('punkt')
from nltk import word_tokenize
from math import log
import re
#Used only to make an argmax
import numpy as np


class Ngram_model:
    def __init__(self, delta = 1, N = 3):
        #Value of N to make Ngram
        self.N = N
        #delta used for smoothing
        self.delta = delta
        self.solutions = ["vient","mentir", "larron", "aidera", "année", "beau", "fous", "femme", 
             "profite", "outils", "fête", "fête", "font", "tard", "on", "nul", 
             "point", "voir", "vend", "muet"
             ]
        self.train_corpus()

    def train_corpus(self):
        corpus = []
        #On construit pour chaque N-gramme un dictionnaire
        for N_gramme in range(self.N):    
            Ngramme_dict = {}
            with codecs.open('proverbes.txt', 'r', 'utf-8') as file:

                #Pour chaque proverbe on tokenize 
                for proverbe in file.readlines(): 
                    
                    #Ajout des tokens de début et fin de phrase
                    tokens = word_tokenize(proverbe)
                    tokens.insert(0, u'<s>')
                    tokens.append(u'</s>')
                    for j in range(0, N_gramme):
                        tokens.insert(0, u'<s>')
                        tokens.append(u'</s>')
                        
                    #Comptage des tuples
                    for i in range(len(tokens) - N_gramme):
                        key = ([x.encode('utf-8') for x in tokens[i:i+N_gramme + 1]])
                        try:
                            Ngramme_dict[tuple(key)] = Ngramme_dict[tuple(key)] + 1
                        except KeyError:
                            Ngramme_dict[tuple(key)] = 1                
            
                #Reduction des comptes des tokens de debut fin phrases
                '''
                TESTER AVEC EST SANS
                '''
                tokenToSuppress = tuple(np.repeat("<s>", N_gramme + 1))
                Ngramme_dict[tokenToSuppress] = 0
                tokenToSuppress = tuple(np.repeat("</s>", N_gramme + 1))
                Ngramme_dict[tokenToSuppress] = 0
            
            #Ajour d'un unknow
            if(N_gramme == 0):
                Ngramme_dict[tuple(['<UNK>'])] = 0                                
            corpus.append(Ngramme_dict)

        self.corpus = corpus
    
    
    def add_delta_smoothing(self):
        #Puisque pour python les dictionnaire sont des objets e t non des variables,
        #on est obligés d'appeler les constructeurs de copie pour assigner et manipuler
        #de nouvelle valeur
        #On ne peut pas faire newCorpus = corpus
        newCorpus = map(lambda x: dict(x), self.corpus)
        delta = self.delta
        
        #Calcul pour le cas 0 (fait à part car on ne peut pas appeler count(w_n-1))
        Ngramme_dict = newCorpus[0]
        N = sum(Ngramme_dict.values())
        V = len(Ngramme_dict.keys()) - 1
        for k in Ngramme_dict.keys():
            Ngramme_dict[k] = float((Ngramme_dict[k]+ delta) * N) / float(N + delta * V) 
         
        #Cas N >= 1
        for i in range(1, len(self.corpus)): 
            Ngramme_dict = newCorpus[i]
            keys = Ngramme_dict.keys()
            
            #Obtention des c* : les comptes apres lissage
            for k in keys:
                #On passe tout en proba et pas en compte reconstitués
                countMinusOne = newCorpus[i-1][tuple(k[0:i])]
                
                Ngramme_dict[k] = (Ngramme_dict[k] + delta) * countMinusOne
                Ngramme_dict[k] = float(Ngramme_dict[k]) / float(countMinusOne + delta * V)
            
            newCorpus[i] = Ngramme_dict
        
        self.corpus = newCorpus
    
    
    #Retourne le nombre d'occurence du tuple si on l'avait stocké en mémoire
    def get_count(self, word_array):
        if(len(word_array) == 0):
            return 1
        #Cas idéal : on a déjà le tuple dans le corpus
        try:
            return self.corpus[len(word_array) -1][tuple(word_array)]
        #Cas le plus probable : le tuple n'est pas dans le corpus et il va falloir
        #déduire sa valeur comme elle aurait été calculée si on l'avait mise dans 
        #le corpus
        except KeyError:
            #Si on n'a pas trouvé le mot dans le corpus des unigrammes, alors
            #on doit utiliser le mot unknow
            if(len(word_array) <= 1):
                return self.corpus[0][tuple(['<UNK>'])]
            #Si on n'a pas trouvé le mot pour un Ngramme autre que unigramme, on va
            #faire le même calcul que pour le lissage pour connaître la valeur qu'il
            #aurait eu si on l'avait stocké
            #Rappel: il est impossible de stocker toutes les combinaisons en mémoire
            else:    
                V = len(self.corpus[0])
                
                #On utilise une récursion car il est possible que l'on soit confronté
                #au même problème pour le N-1gramme, à savoir un tuple jamais rencontré
                count_minus1 = self.get_count(word_array[:-1])
                
                return float(self.delta * count_minus1) / float(count_minus1  + self.delta * V) 
        return 1
    
    
    
    def compute_probs(self, word_array, N = 3):
        probs = []
        
        #On ne veut pas utiliser un  quadrigramme si on n'a que trois mots
        if(N > len(self.corpus)):
            return 0
    
        #Tokenization de la phrase d'entrée et ajout des marqueurs de début et fin de phrase
        tokens = word_tokenize(word_array)
        for j in range(N - 1):
            tokens.insert(0, '<s>')
            tokens.append('</s>')
        for i in range(len(tokens) - (N-1)):        
            #Tuple par tuple, on appelle la fonction get_count
            word_tuple = tokens[i:(i+N)]
            count = self.get_count(word_tuple)
        
            proba = count / self.get_count(tokens[i:(i+N-1)])
            probs.append(float(proba))
            #print(proba)
    
        #On renvoie pour chaque probabilité son log
        probs = map(lambda x: log(x,2), probs)
        log_proba = sum(probs)    
        try:
            crossEntropy = -1 * log_proba / float(len(word_tokenize(word_array)))
            perplexity = 2 ** crossEntropy
    
        #Il peut arriver que log_proba soit trop petit pour etre elevé à une puissance négative
        #Cela donnerait une valeur infini
        except ZeroDivisionError:
            perplexity = 99999999999999999        
        
        
        #print('------')
        #print(perplexity, 'perplexity')
        return sum(probs), perplexity
        
    def test_models(self):
        predictions = np.array([])
        for N_gramme in range(self.N):
            print(N_gramme + 1, self.delta)
            with codecs.open('test1.txt', 'r', 'utf-8-sig') as file:
                for line in file.readlines(): 
                    
                    proverbe = re.findall('"(.+)":', line)
                    
                    propositions = re.findall('\"((\w|à|\xe2|û|ï|\xea|\xe9|\xee|\xe8)+)\"(,|\])', line)
                    propositions = map(lambda x: x[0], propositions)    
                    
    
                    complete_proverbs = map(lambda x: proverbe[0].encode('utf-8').replace('***', x.encode('utf-8')), propositions)
                    probs = map(lambda x: self.compute_probs(x, N_gramme + 1)[0], complete_proverbs)
                    #print(probs)
                    probs = np.array(probs)
                    try :
                        print('---------------------')
                        '''
                        print(proverbe[0])
                        print(propositions)
                        '''
                        print(proverbe[0].encode('utf-8').replace('***', propositions[np.argmax(probs)].encode('utf-8')))
                        predictions = np.append(predictions, [propositions[np.argmax(probs)].encode('utf-8')])
                    except IndexError:
                        print('NONE')
        

        predictions = predictions.reshape((self.N, int(float(len(predictions))/self.N)))
        for i in range(self.N):
            score = 0
            for j in range(predictions.shape[1]):
                if (predictions[i, j] == self.solutions[j]):
                    score = score + 1
            print("N = " + str(i + 1),float(score) / 20)
            
            
            
if __name__ == '__main__':
    predictor = Ngram_model(0.01, 7)
    predictor.add_delta_smoothing()
    predictor.test_models()
