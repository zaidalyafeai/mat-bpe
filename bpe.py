import nltk
from collections import Counter
import re
import morfessor
import random 
from farasa.segmenter import FarasaSegmenter
import pickle 
from utils import *

nltk.download('punkt')


class bpe:
  def __init__(self, vocab_size = 100, verbose = False, morph = False, do_init_merge = False, prob = 0, lang = 'en'):
    self.do_init_merge = do_init_merge
    self.vocab = [PAD, UNK, SOW, EOW] 
    self.morph = morph     
    self.merges = []
    self.prob = prob
    self.vocab_size = vocab_size
    self.verbose = verbose
    self.lang = lang
    self.name = f'bpe-lang-{lang}'
    if self.morph:
      self.name += '-morph'
      if lang == 'en':
        io = morfessor.MorfessorIO()
        self.segmenter = io.read_binary_model_file('/content/drive/MyDrive/DISS/morfessor.bin')
      elif lang == 'ar':
        self.segmenter = FarasaSegmenter() 
    self.name += f'-prob-{prob}'

  def extract_affixes(self, t):
    affixes = set()
    for word in t.split(' '):
      if len(word) == 0:
        continue
      if self.lang == 'en':
        morphemes = self.segmenter.viterbi_segment(word)[0]
      elif self.lang == 'ar':
        morphemes = self.segmenter.segment(word)
        morphemes = morphemes.split('+')
      
      if len(morphemes) == 1:
        continue
      
      max_len = max([len(morpheme) for morpheme in morphemes])
      for morpheme in morphemes:
        affix = morpheme
        if len(affix) < max_len:
          if word.startswith(affix):
            affix = SOW+affix
          elif word.endswith(affix):
            affix = affix+EOW
          affixes.add(affix)
    return affixes

  def train(self, text = None, file = None):
    if text:
      t = text
    if file:
      t = open(file, 'r').read()

    self.corpus = process(t)
    self.vocab += [char for char in set(t.replace(' ', ''))]
    self.vocab = self.vocab[:self.vocab_size]

    if self.morph:
      print('extracting affixes ...')
      affixes = self.extract_affixes(t)
      # print(affixes)
      init_merges = generate_merges(affixes)
      self.vocab += [('').join(merge) for merge in init_merges]
      self.vocab = self.vocab[:self.vocab_size]
      self.merges += init_merges
    
    if self.do_init_merge:
      while True: 
        best_pair = None 
        pairs = get_pairs(self.corpus)
        for pair in self.merges:
          if pair in pairs:
            best_pair = pair
            break

        if best_pair is None:
          break 

        self.vocab.append(('').join(best_pair))
        self.corpus = merge(self.corpus, best_pair)
    
    while True:
      grams_count = get_pairs(self.corpus).most_common()

      # stop conditions
      if len(grams_count) == 0:
        print('no more bigrams to merge')
        break
      if len(self.vocab) > self.vocab_size:
        print('vocab size reached')
        break

      # stochastic seeds 
      if self.prob > random.random():
        idx = random.randint(0, len(grams_count) - 1)
        best_pair = grams_count[idx][0]
      else:
        best_pair = grams_count[0][0]

      # build vocab and merges
      self.vocab.append(('').join(best_pair))
      self.corpus = merge(self.corpus, best_pair)
      self.merges.append(best_pair)
 
      if self.verbose:
        print(self.corpus)

  def _encode_word(self, word):
    tokens = self._tokenize_word(word)
    return [self.vocab.index(token) for token in tokens]

  def _encode(self, sentence, out_len = None):
    output = [self._encode_word(word) for word in sentence.split(' ')] 
    output = [item for sublist in output for item in sublist]

    
    if out_len is None:
      return output
    else:
      if out_len > len(output):
        return output + [self.vocab.index(PAD)] * max(out_len - len(output), 0)
      else:
        return output[:out_len]

  def encode(self, text, out_len = None):
    if type(text) is str:
      return self._encode(text, out_len = out_len)
    elif type(text) is list:
      return [self._encode(stmt, out_len = out_len) for stmt in text]
    else:
      raise('Error, not familiar type')

  def decode(self, ids):
    if type(ids[0]) is list:
      output = []
      for inst in ids:
        output.append([self.vocab[id] for id in inst])
      return output
    else:
      return [self.vocab[id] for id in ids]

  def tokenize(self, sentence):
    return [self._tokenize_word(word) for word in sentence.split(' ')]

  def _tokenize_word(self, t):
    t = process(t)
    t = (' ').join([char if char in self.vocab else UNK for char in t.split(' ')])
    while True:
      pairs = get_pairs(t)
      best_pair = None 

      for pair in self.merges:
        if pair in pairs:
          best_pair = pair
          break

      if best_pair is None:
        break 
      t = merge(t, best_pair)
    return t.split(' ')

  def save(self, path):
    with open(path, 'wb') as handle:
      pickle.dump([self.vocab, self.merges], handle, protocol=pickle.HIGHEST_PROTOCOL)

  def load(self, path):
    with open(path, 'rb') as handle:
      self.vocab, self.merges = pickle.load(handle)