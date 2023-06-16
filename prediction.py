import pickle as pkl

import pandas as pd
import numpy as np


import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import joblib
import streamlit as st

def predict(inp):
   
  class preprocessor:
    def __init__(self, sentences):
      self.sen = sentences

    def sent_tokenizer(self,sent):

      sent = nltk.sent_tokenize(sent)

      for i in range(len(sent)):

        sent[i] = sent[i][:len(sent[i])-1]
      sent = ' '.join(sent)

      return sent

    def tokenize_it(self, sen):
      # instantiate tokenizer class
      sen_new = nltk.word_tokenize(sen)
      return sen_new

    def remove_stopW_and_punc(self, sen_tokens):
      stopwords_english = stopwords.words('english')
      sen_clean = []

      for word in sen_tokens: # Go through every word in your tokens list
          if (word not in stopwords_english and  # remove stopwords
              word not in string.punctuation):  # remove punctuation
              sen_clean.append(word)
      return sen_clean


    def sen_stemmer(self, sen_clean):
      # Instantiate stemming class
      stemmer = PorterStemmer() 

      # Create an empty list to store the stems
      sen_stem = [] 

      for word in sen_clean:
          stem_word = stemmer.stem(word)  # stemming word
          sen_stem.append(stem_word)  # append to the list
      
      return sen_stem

    # Now we setup a controller which initiates each mehtod of the above declared 
    # class in the correct chronological order.

    def controller(self, sen):
        temp_sen = self.sent_tokenizer(sen)
        temp_sen = self.tokenize_it(temp_sen)
        temp_sen = self.remove_stopW_and_punc(temp_sen)
        temp_sen = self.sen_stemmer(temp_sen)

        return temp_sen

    def run(self):
      processed_sens = []
      while self.sen:
        sen = self.sen.pop(0)
        processed_sen = self.controller(sen)
        processed_sens.append(processed_sen)
      processed_sens = np.array(processed_sens)
      return processed_sens


  
  model = joblib.load("model.sav")

  Vectorizer = joblib.load("Vectorizer.sav")

  #creating a function for predicting the category
    
  #write your input code here
  # inp = input("write a question:")
  #write your input code here


  lst = []
  lst.append(inp)
  processed_sentences_predict = preprocessor(lst).run()

  temp = []

  for i in range(len(processed_sentences_predict)):
    temp.append(' '.join(processed_sentences_predict[i]))

  X_predict = Vectorizer.transform(np.array(temp))

  prediction = model.predict(X_predict.toarray())
  prediction_list = prediction.tolist()

  rev_ref_dict = {0:'UPSC',1:'Research',2:'Corporate'}
  result = rev_ref_dict[prediction[0]]
  return result

    
  # #write your input code here
  # inp = input("write a question:")
  # #write your input code here


  # lst = []
  # lst.append(inp)
  # processed_sentences_predict = preprocessor(lst).run()

  # temp = []

  # for i in range(len(processed_sentences_predict)):
  #   temp.append(' '.join(processed_sentences_predict[i]))

  # X_predict = Vectorizer.transform(np.array(temp))

  # prediction = model.predict(X_predict.toarray())
  # prediction_list = prediction.tolist()

  # rev_ref_dict = {0:'UPSC',1:'Research',2:'Corporate'}
  # result = rev_ref_dict[prediction[0]]

  # print(result)

    


