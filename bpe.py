import nltk
from collections import Counter
import re
import morfessor
import random 
from farasa.segmenter import FarasaSegmenter
import pickle 


nltk.download('punkt')
EOW = '</w>'
UNK = '<unk>'
PAD = '<pad>'

def split_affixes(affixes):
  """
  splits affixes based on approximations 
  returns: a list of tuples ed</w> => [(d, </w>), (e, d</w>)]
  """
  merges = []
  for affix in affixes:
    dir = 'mid'
    if EOW in affix:
      dir = 'rtl'
      affix = affix.replace(EOW, '')
      chars = list(affix)+[EOW]
    else:
      chars = list(affix)
    curr_merge = []

    while len(chars) > 1:
      if dir == 'mid':
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
  """
  get pairs of bigrams with frequency 
  returns: a counter of the bigrams in the corpus
  """
  grams = Counter()
  tokens = t.split(' ') 
  bigrams = list(nltk.bigrams(tokens))
  
  for bigram in bigrams:
    if bigram[0] == UNK:
      continue
    if not bigram[0].endswith(EOW): # don't combine across words
      grams[bigram] += 1
  return grams
  
class bpe:
  """
  Tokenizer main class
  """
  def __init__(self, vocab_size = 100, verbose = False, morph = False, prob = 0, lang = 'en', lower_case = True):
    self.vocab = [PAD, UNK, EOW] 
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
        self.segmenter = io.read_binary_model_file('morfessor.bin')
      elif lang == 'ar':
        self.segmenter = FarasaSegmenter() 
    self.name += f'-prob-{prob}'
    self.lower_case = lower_case 

  def extract_affixes(self, t):
    """
    Extract all affixes given a spcific language using the segmenter
    returns: a list of affixes wanted => ['ed</w>']
    """
    affixes = set()
    if self.lang == 'en':
      for word in t.split(' '):
        if len(word) == 0:
          continue
        morphemes = self.segmenter.viterbi_segment(word)[0]
        
        if len(morphemes) == 1:
          continue
        
        max_len = max([len(morpheme) for morpheme in morphemes])
        for morpheme in morphemes:
          affix = morpheme
          if len(affix) < max_len:
            if word.endswith(affix):
              affix = affix+EOW
            affixes.add(affix)

    if self.lang == 'ar':
      for word in self.segmenter.segment(t).split(' '):
        if len(word) == 0:
          continue
        
        morphemes = word.split('+')
        
        if len(morphemes) == 1:
          continue
        
        
        max_len = max([len(morpheme) for morpheme in morphemes])
        for morpheme in morphemes:
          affix = morpheme
          if len(affix) < max_len: #exclude the main stem from the list of affixes 
            if word.endswith(affix):
              affix = affix+EOW
            affixes.add(affix)

    return affixes

  def merge(self, t, bigram):
    """
    join a bigram in a given text corpus
    """
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

  def preprocess(self, t):
    """
    split text corpus into characters: 
    returns: split format hello => <w> h e l l o </w>
    """
    if self.lower_case:
      t = t.lower()
    t = (' ').join([(' ').join(list(word))+f' {EOW}' for word in t.split(' ')])
    return t

  def train(self, text = None, file = None):
    """
    train on either a plain text or a file 
    """
    if text:
      t = text
    elif file:
      t = open(file, 'r').read()
    else:
      raise("Must use corpus using plain text or a file")

    self.corpus = self.preprocess(t)
    self.vocab += [char for char in set(t.replace(' ', ''))]

    if len(self.vocab) > self.vocab_size:
        raise Exception('Minimum vocab size is ', len(self.vocab))

    if self.morph:
      print('extracting affixes ...')
      affixes = self.extract_affixes(t)
      init_merges = split_affixes(affixes)

      for merge in init_merges:
        if len(self.vocab) >= self.vocab_size:
          break
        self.vocab.append(('').join(merge))
        self.merges.append(merge)
      
      while True:
        pair_to_merge = None 
        pairs = get_pairs(self.corpus)
        for pair in self.merges:
          if pair in pairs:
            pair_to_merge = pair
            break
        
        if pair_to_merge:
          self.corpus = self.merge(self.corpus, pair)
        else:
          break
        

    step = 0 
    while True:
      grams_count = get_pairs(self.corpus).most_common()

      # stop conditions
      if len(grams_count) == 0:
        print('no more bigrams to merge')
        break
      if len(self.vocab) >= self.vocab_size:
        print('vocab size reached')
        break

      # randomly choose some grams  
      if self.prob > random.random():
        idx = random.randint(0, len(grams_count) - 1)
        best_pair = grams_count[idx][0]
      else:
        best_pair = grams_count[0][0]

      # build vocab and merges
      self.vocab.append(('').join(best_pair))
      self.corpus = self.merge(self.corpus, best_pair)
      self.merges.append(best_pair)
 
      if self.verbose:
        print(f'step: {step}, merges: {self.merges}, vocab: {self.vocab}')
      step += 1

  def _encode_word(self, word):
    """
    encode a single word
    """
    tokens = self._tokenize_word(word)
    return [self.vocab.index(token) for token in tokens]

  def _encode_sentence(self, sentence, out_len = None):
    """
    encode a senteces
    """
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
    """
    encode a text corpus
    """
    if type(text) is str:
      return self._encode_sentence(text, out_len = out_len)
    elif type(text) is list:
      return [self._encode_sentence(stmt, out_len = out_len) for stmt in text]
    else:
      raise('Error, not familiar type')

  def decode(self, ids):
    """
    Decode a list of ids
    """
    if type(ids[0]) is list:
      output = []
      for inst in ids:
        output.append([self.vocab[id] for id in inst])
      return output
    else:
      return [self.vocab[id] for id in ids]

  def tokenize(self, sentence):
    """
    tokenize a sentence
    """
    return [self._tokenize_word(word) for word in sentence.split(' ')]

  def _tokenize_word(self, t):
    """
    tokenize a single word 
    """
    t = self.preprocess(t)
    t = (' ').join([char if char in self.vocab else UNK for char in t.split(' ')])
    
    while True:
      pairs = get_pairs(t)
      best_pair = None 

      for pair in self.merges:
        if pair in pairs:
          best_pair = pair
          break

      # stopping criteria no more merges 
      if best_pair is None:
        break 
      t = self.merge(t, best_pair)
    return t.split(' ')

  def save(self, path):
    """
    save merges using file name 
    """
    with open(f'{path}/tok.model', 'wb') as handle:
      pickle.dump([self.vocab, self.merges], handle, protocol=pickle.HIGHEST_PROTOCOL)

  def load(self, path):
    with open(f'{path}/tok.model', 'rb') as handle:
      self.vocab, self.merges = pickle.load(handle)
