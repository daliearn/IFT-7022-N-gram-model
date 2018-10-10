#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
'''
with open('ingredients.txt', 'r') as file:
    ingredient = file.read().replace('\n', '')
''' 

#match = re.findall("^[0-9]+(,[0-9]+)?( tasse| cuillère(s)?| oz| mL| g| c\.à\.s| c\.à\.c)?")    

#(^[0-9]+(,[0-9]+)?( tasse| cuillère(s)?| oz| mL| g| c\.à\.s| c\.à\.c)?)( de)? ((.)+$)


#^(.+) (de |d’)(.+)
#^((.+)\(.+\)) (.+)

#((^[0-9]+|½)((,|\.|\/)[0-9]+)?( tasse| feuille(s)?| cuillère(s)?( à soupe| à café)?| pinte(s)?| oz| mL| g| c\.à\.s| c\.à\.c| c. à thé)?( \(.+\))?)(de)? ((.)+$)


def get_ingredients(string):
    match = re.findall("(^[0-9]+((,|.)[0-9]+)?( tasse| feuille(s)?| cuillère(s)?| pinte(s)?| oz| mL| g| c\.à\.s| c\.à\.c)?)( de)? ((.)+$)", string)
    if(len(match) > 0):
        return match[0][0], match[0][8]
    else:        
        return "",""
    
score = 0
count = 0    
    
with open('ingredients.txt', 'r') as file, open('ingredients_solutions.txt') as solutions:
    for line, solution in zip(file.readlines(), solutions.readlines()):
        qty, ingredient = get_ingredients(line) 
        answ = line.replace('\n', '') + "   QUANTITE:" + qty + "   INGREDIENT:" + ingredient + '\n'
        print(answ)
        print(solution)
        print('----------------------')        
        print("===> ", answ == solution)
        if (answ == solution):
            score = score + 1
        count = count + 1            

print(score, count)
        