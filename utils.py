import nltk
from collections import Counter
import re
import random 

def generate_merges(affixes):
  merges = []
  for affix in affixes:
    dir = 'mid'
    if EOW in affix:
      dir = 'rtl'
      affix = affix.replace(EOW, '')
      chars = list(affix)+[EOW]
    elif SOW in affix:
      dir = 'ltr'
      affix = affix.replace(SOW, '')
      chars = [SOW] + list(affix)
    else:
      chars = list(affix)
    curr_merge = []
    while len(chars) > 1:
      if dir == 'ltr' or dir == 'mid':
        curr_merge.append((chars[0], chars[1]))
        chars[1] = chars[0]+chars[1]
        chars = chars[1:]
      else:        
        curr_merge.append((chars[-2], chars[-1]))
        chars[-2] = chars[-2]+chars[-1]
        chars = chars[:-1]

    for merge in curr_merge:
      if merge not in merges:
        merges.append(merge)
  return merges
  
def get_pairs(t):
  grams = Counter()
  tokens = t.split(' ') 
  bigrams = list(nltk.bigrams(tokens))
  
  for bigram in bigrams:
    if bigram[0] == UNK:
      continue
    if not bigram[0].endswith(EOW): # don't combine across words
      grams[bigram] += 1
  return grams
  
def process(t):
  t = (' ').join([f'{SOW} '+(' ').join(list(word))+f' {EOW}' for word in t.split(' ')])
  return t

def merge(t, bigram):
  tokens = t.split(' ')
  new_tokens = []
  i = 0 
  while i < len(tokens):
    if ('').join(tokens[i:i+2]) == ('').join(bigram):
      new_tokens.append(('').join(bigram))
      i += 1
    else:
      new_tokens.append(tokens[i])
    i += 1
  return (' ').join(new_tokens)
