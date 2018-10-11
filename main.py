#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
'''
with open('ingredients.txt', 'r') as file:
    ingredient = file.read().replace('\n', '')
''' 

def get_ingredients(string):
    #La regex des quantités est très longues du fait du nombre énorme d'unités    
    qty = re.findall('^(?:\d+|½|[U|u]n[e]?)(?:(?:,|.)[0-9]+)?(?: tasse[s]? | lamelle[s]? | noix | bouquet | [Rr]ondelle[s]? | Bouquet | morceau | feuille[s]? | cuillère[s]?(?: à soupe| à café)? | pinte[s]? | oz | mL | ml | g | c[ ]?\.[ ]?à[ ]?[\.]?[ ]?s[\.]? | c[ ]?\.[ ]?à[ ]?[\.]?[ ]?c[\.]? | c\.[ ]?à (?:thé|soupe) | pincée[s]? | lb | tranche[s]? | cl | cL | verres | tronçon[s]? | botte | boîte | kg | enveloppe | gallon | pièce[s]? | gousse[s]? )?(?: )?(?:\(.+\))?',string)    
    
    #traitement de ce qu'on a matché (suppression d'un éventuel espace à la fin)
    if(len(qty) > 0):
        if(qty[0][-1] == ' '):
            qty = qty[0][:-1]
        else:    
            qty = qty[0]
    else:        
        qty = ""
    
    
    #On tente avec la chaîne entière (pour les ingredients sans quantité)
    ing = re.findall('^[^0-9½|[Uu]n[e]?].+', string)
    
    #Si on a rien, on tente une autre regex
    if(len(ing) == 0):
        #Pour éviter la multiplication de groupes de capture, on enlève la string que l'on vient de trouver
        string = string[len(qty): len(string) - 1]
        ing = re.findall("(:? )?(:?de |d[’'])?(.+)(:?au goût)?"  , string)
    
        #Suppression de caractères génants
        if(len(ing) > 0):
            if(ing[0][2][-1] == ' ' or ing[0][2][-1] == '\n'):
                ing = ing[0][2][:-1]
            else:
                ing = ing[0][2]
        else:        
            ing = ""
    #Si on avait déjà une regex on a juste a supprimer les caractères génants
    else:
        if(ing[0][-1] == ' ' or ing[0][-1] == '\n'):
            ing = ing[0][:-1]
        else:
            ing = ing[0]
    
        
    #Enlève les adjectifs de la forme ées (ne change pas les perfs de perf)
    if (len(re.findall("(.+)(:? \w+é\w{0,2}[ |$])", ing)) > 0):
        ing = re.findall("(.+)(:? \w+é\w{0,2}[ |$])", ing)[0][0]
    
    
    #Enlève ce qu'il y a après la virgule (ne change pas les perfs de perf)
    if (len(re.findall("^(.+),", ing)) > 0):
        ing = re.findall("^(.+),", ing)[0]
    
    
    
    return qty, ing
        

if __name__ == '__main__':
    score = 0
    score_ing = 0
    score_qte = 0
    count = 0    
    
    with open('ingredients.txt', 'r') as file, open('ingredients_solutions.txt') as solutions:
        for line, solution in zip(file.readlines(), solutions.readlines()):
            qty, ingredient = get_ingredients(line) 
            
            try:
                answ_qty = re.findall('QUANTITE:(.+)   ', solution)[0]
            except IndexError:
                answ_qty = ""
                
            try:
                answ_ing = re.findall('INGREDIENT:(.+)$', solution)[0]
            except IndexError:
                answ_ing = ""
    
            
            if (answ_qty == qty):
                score_qte = score_qte + 1
            else:
                print(qty, answ_qty)
                pass
            if (answ_ing == ingredient):
                score_ing = score_ing + 1
            else:
                #print(ingredient, answ_ing)
                pass
    
            if (answ_qty == qty and answ_ing == ingredient):
                score = score + 1
            
            count = count + 1         
            
    print('qte', score_qte, count)
    print('ing', score_ing, count)
    print('both exact', score, count)
           