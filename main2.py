#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
#nltk.download('punkt')
from nltk import word_tokenize
from math import log, exp
import re
#Used only to make an argmax
import numpy as np

def train_corpus():
    corpus = []
    #On construit pour chaque N-gramme un dictionnaire
    for N_gramme in range(3):    
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
        
        #Ajour d'un unknow
        if(N_gramme == 0):
            Ngramme_dict[tuple(['<UNK>'])] = 0
        corpus.append(Ngramme_dict)

    return corpus


def add_delta_smoothing(corpus, delta = 1):
    Ngramme_dict = corpus[0]
    N = sum(Ngramme_dict.values())
    V = len(Ngramme_dict.keys()) - 1
    for k in Ngramme_dict.keys():
        Ngramme_dict[k] = float((Ngramme_dict[k]+ delta) * N) / float(N + delta * V) 
     
        
    for i in range(1, len(corpus)):    
        #V = word_number ** (i+1)
        V = len(corpus[0]) - 1
        Ngramme_dict = corpus[i]
        keys = Ngramme_dict.keys()
        
        #Obtention des c* : les comptes apres lissage
        for k in keys:
            #On passe tout en proba et pas en compte reconstitués
            #Ngramme_dict[k] = (Ngramme_dict[k] + delta)
            N = corpus[i-1][tuple(k[0:i])]
            
            Ngramme_dict[k] = (Ngramme_dict[k] + delta) * N
            Ngramme_dict[k] = float(Ngramme_dict[k]) / float(N + delta * V)
        
        corpus[i] = Ngramme_dict
    
    return corpus

#Retourne le nombre d'occurence du tuple si on l'avait stocké en mémoire
def get_count(word_array, corpus, delta = 1):
    if(len(word_array) == 0):
        return 1
    #Cas idéal : on a déjà le tuple dans le corpus
    try:
        return corpus[len(word_array) -1][tuple(word_array)]
    #Cas le plus probable : le tuple n'est pas dans le corpus et il va falloir
    #déduire sa valeur comme elle aurait été calculée si on l'avait mise dans 
    #le corpus
    except KeyError:
        #Si on n'a pas trouvé le mot dans le corpus des unigrammes, alors
        #on doit utiliser le mot unknow
        if(len(word_array) <= 1):
            return corpus[0][tuple(['<UNK>'])]
        #Si on n'a pas trouvé le mot pour un Ngramme autre que unigramme, on va
        #faire le même calcul que pour le lissage pour connaître la valeur qu'il
        #aurait eu si on l'avait stocké
        #Rappel: il est impossible de stocker toutes les combinaisons en mémoire
        else:    
            V = len(corpus[0])
            
            #On utilise une récursion car il est possible que l'on soit confronté
            #au même problème pour le N-1gramme, à savoir un tuple jamais rencontré
            count_minus1 = get_count(word_array[:-1], corpus, delta)
            
            #return float(delta) / float(count_minus1  + delta * V)
            return float(delta * count_minus1) / float(count_minus1  + delta * V) 
    return 1



def compute_probs(word_array, corpus, N, delta = 1):
    probs = []
    #On ne veut pas utiliser un  quadrigramme si on n'a que trois mots
    if(N > len(corpus)):
        return 0

    #Tokenization de la phrase d'entrée et ajout des marqueurs de début et fin de phrase
    tokens = word_tokenize(word_array)
    for j in range(N - 1):
        tokens.insert(0, '<s>')
        tokens.append('</s>')
    print('------')
    for i in range(len(tokens) - (N-1)):        
        #Tuple par tuple, on appelle la fonction get_count
        word_tuple = tokens[i:(i+N)]
        count = get_count(word_tuple, corpus, delta)
    
        proba = count / get_count(tokens[i:(i+N-1)], corpus, delta)
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
    
    
    print(perplexity, 'perplexity')
    return sum(probs), perplexity
    
def test_models(corpus, delta = 1):
    for N_gramme in range(3):
        print(N_gramme + 1, delta)
        with codecs.open('test1.txt', 'r', 'utf-8-sig') as file:
            for line in file.readlines(): 
                
                proverbe = re.findall('"(.+)":', line)
                
                propositions = re.findall('\"((\w|à|\xe2|û|ï|\xea|\xe9|\xee|\xe8)+)\"(,|\])', line)
                propositions = map(lambda x: x[0], propositions)    
                
                
                probs = map(lambda x: proverbe[0].encode('utf-8').replace('***', x.encode('utf-8')), propositions)
                
                probs = map(lambda x: compute_probs(x, corpus, N_gramme + 1, delta)[0], probs)
                #print(probs)
                
                probs = np.array(probs)
                try :
                    
                    print('---------------------')
                    '''
                    print(proverbe[0])
                    print(propositions)
                    '''
                    print(proverbe[0].encode('utf-8').replace('***', propositions[np.argmax(probs)].encode('utf-8')))
                    print('---------------------')
                except IndexError:
                    print('NONE')
        
                
    
def megaTest():
    corpus = train_corpus()
    for delta in [0.1, 0.3, 0.4, 0.7, 1, 2]:
        corpus2 = add_delta_smoothing(corpus, delta)    
        test_models(corpus2, delta)
    
if __name__ == '__main__':
    corpus = train_corpus()
    corpus2 = add_delta_smoothing(corpus)
    test_models(corpus)
    

