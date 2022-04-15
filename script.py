# -*- coding: utf-8 -*-
"""Copy of transformer ar2en (LDC).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1du52gJa1S_tffOKkITdMoMKED_w3xwTI

# Transformer model for language understanding
"""
import os 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import GRU, Embedding, Dense, Input, Dropout, Bidirectional
import tkseem as tk
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import tnkeeh as tn
from bpe import *
import argparse
import logging 
import sys

parser = argparse.ArgumentParser(description='Args')
parser.add_argument('--tok', type=int)
parser.add_argument('--run', type=int)
parser.add_argument('--vocab_size', type=int)
parser.add_argument('--dir', type=str)
parser.add_argument('--max_word_tokens', type=int)
parser.add_argument('--max_char_tokens', type=int)

args = parser.parse_args()
print(args)
"""## Setup input pipeline"""

MAX_TOKENS = args.max_word_tokens

train_text = open('/content/train_data.txt', 'r').read().splitlines()
train_lbls = [int(lbl) for lbl in open('/content/train_labels.txt', 'r').read().splitlines()]
valid_text = open('/content/valid_data.txt', 'r').read().splitlines()
valid_lbls = [int(lbl) for lbl in open('/content/valid_labels.txt', 'r').read().splitlines()]
test_text = open('/content/test_data.txt', 'r').read().splitlines()
test_lbls = [int(lbl) for lbl in open('/content/test_labels.txt', 'r').read().splitlines()]

assert len(train_text) == len(train_lbls)
assert len(test_text) == len(test_text)

def tokenize_data(tokenizer, vocab_size = 10000):
  train_data = tokenizer.encode(train_text, out_len=MAX_TOKENS)
  valid_data = tokenizer.encode(valid_text, out_len=MAX_TOKENS)
  test_data = tokenizer.encode(test_text, out_len=MAX_TOKENS)
  return tokenizer, (train_data, train_lbls), (valid_data, valid_lbls), (test_data, test_lbls)

def create_dataset(train_data, valid_data, test_data, batch_size = 256, buffer_size = 50000):
  train_dataset = tf.data.Dataset.from_tensor_slices(train_data).shuffle(buffer_size)
  train_dataset = train_dataset.batch(batch_size, drop_remainder=True)

  valid_dataset = tf.data.Dataset.from_tensor_slices(valid_data)
  valid_dataset = valid_dataset.batch(batch_size)

  test_dataset = tf.data.Dataset.from_tensor_slices(test_data)
  test_dataset = test_dataset.batch(batch_size)
  return train_dataset, valid_dataset, test_dataset


def create_model(vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 128))
    model.add(Bidirectional(GRU(units = 256, return_sequences = True)))
    model.add(Bidirectional(GRU(units = 256)))
    model.add(Dense(1, activation = 'sigmoid'))
    return model

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False,
              reduction = 'none')

def loss_function(real, pred):
  loss_ = loss_object(real, pred)
  return tf.reduce_mean(loss_)

def accuracy_function(real, pred):
  result = tf.equal(tf.squeeze(real),tf.squeeze(tf.cast(tf.round(pred), tf.int32))) 
  return tf.reduce_mean( tf.cast(result, tf.float32))

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

valid_loss = tf.keras.metrics.Mean(name='valid_loss')
valid_accuracy = tf.keras.metrics.Mean(name='valid_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.Mean(name='test_accuracy')


train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
]

# @tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    
  with tf.GradientTape() as tape:
    predictions = model(inp)
    loss = loss_function(tar, predictions)

  gradients = tape.gradient(loss, model.trainable_variables)    
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  
  train_loss(loss)
  train_accuracy(accuracy_function(tar, predictions))

# @tf.function(input_signature=train_step_signature)
def valid_step(inp, tar):
  predictions = model(inp)
  loss = loss_function(tar, predictions)
  
  valid_loss(loss)
  valid_accuracy(accuracy_function(tar, predictions))

# @tf.function(input_signature=train_step_signature)
def test_step(inp, tar):
  predictions = model(inp)
  loss = loss_function(tar, predictions)
  
  test_loss(loss)
  test_accuracy(accuracy_function(tar, predictions))

def train(epochs = 30, verbose = 0):
  best_score = 10
  for epoch in range(epochs):
    start = time.time()
    
    train_loss.reset_states()
    train_accuracy.reset_states()

    valid_loss.reset_states()
    valid_accuracy.reset_states()
    
    # inp -> portuguese, tar -> english
    for (batch, (inp, tar)) in enumerate(train_dataset):
      train_step(inp, tar)
      
      if batch % 500 == 0 and verbose:
        print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
            epoch + 1, batch, train_loss.result(), train_accuracy.result()))
        
      

    for (batch, (inp, tar)) in enumerate(valid_dataset):
      valid_step(inp, tar)
      
      
    print ('Epoch {} Train Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                  train_loss.result(), 
                                                  train_accuracy.result()))
    
    print ('Epoch {} Valid Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                  valid_loss.result(), 
                                                  valid_accuracy.result()))
    
    print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    if  valid_loss.result() < best_score:
      best_score = valid_loss.result()
      ckpt_save_path = os.path.basename(os.path.normpath(ckpt_manager.save()))
      
      print ('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))

def evaluate_test(test_dataset):
  test_loss.reset_states()
  test_accuracy.reset_states()
  for (batch, (inp, tar)) in enumerate(test_dataset):
      test_step(inp, tar)

  return test_loss.result().numpy(), test_accuracy.result().numpy()

results = {}

BATCH_SIZE = 256
vocab_size = int(args.vocab_size)

checkpoint_dir = f'{args.dir}/ckpts/'
tokenizers = [bpe(vocab_size, lang = 'ar'), bpe(vocab_size, lang = 'ar', morph = True)]

#updated to add the vocab size as a directory
accs_path = f"{checkpoint_dir}/vocab_size_{vocab_size}/accuracies.json"

if os.path.isfile(accs_path):
  accs = defaultdict(
      list,
      json.load(
      open(
          accs_path,
          mode='r'),
      )
  )
else:
  accs = defaultdict(list)

i = int(args.tok)
j = int(args.run)


# start from the second tok. The first one has completed
tokenizer = tokenizers[i]
name = tokenizer.name
import pickle
if j != len(accs[name]):
  print('This is run already finished')
else:
  tok_dir = f"{checkpoint_dir}/vocab_size_{vocab_size}/{name}"
  if os.path.isfile(f"{tok_dir}/tok.model"):
    print('loading pretrained tokenizer')
    tokenizer.load(f"{tok_dir}")
    with open(f'tok.data', 'rb') as handle:
      train_data, valid_data, test_data = pickle.load(handle)
  else:
    print('training tokenizer from scratch')
    tokenizer.train(file = '/content/train_data.txt')
    tokenizer, train_data, valid_data, test_data = tokenize_data(tokenizer, vocab_size = vocab_size)
    
    with open(f'tok.data', 'wb') as handle:
      pickle.dump([train_data, valid_data, test_data], handle, protocol=pickle.HIGHEST_PROTOCOL)

  train_dataset, valid_dataset, test_dataset = create_dataset(train_data, valid_data, test_data, batch_size = BATCH_SIZE)

  start = time.time()

  optimizer = tf.keras.optimizers.Adam()
  model = create_model(vocab_size)
  # create checkpoint object
  checkpoint = tf.train.Checkpoint(
          optimizer=optimizer,
          model=model
  )
  ckpt_manager = tf.train.CheckpointManager(
      checkpoint,
      f'{checkpoint_dir}/vocab_size_{vocab_size}/{name}/run_{j}',
      max_to_keep=1,
      checkpoint_name='ckpt',
  )

  print(f'run: {j}')
  train(epochs = 20)
  if not os.path.isfile(f"{tok_dir}/tok.model"):
    tokenizer.save(f"{tok_dir}")
  # restore best model
  checkpoint.restore(ckpt_manager.latest_checkpoint)
  _, test_score = evaluate_test(test_dataset)
  print('results on test score is:',test_score)

  accs[name].append(str(test_score))
  json.dump(accs,open(accs_path,mode='w'),indent=4,)
