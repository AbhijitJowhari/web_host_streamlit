import pandas as pd
import numpy as np
import pickle


import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import string

data = pd.read_csv(r'data\user_data.csv')
data.drop(104,axis = 0,inplace = True)
data.drop(['Timestamp','C1'],axis = 1,inplace =True)
data.dropna(inplace = True)
lst_1 = list(data['Questions '])
lst_2 = list(data['Category'])
data = pd.DataFrame({'Questions':lst_1,'Category':lst_2},index = list(range(117)))
X = np.array(data['Questions'])
y = np.array(data['Category'])

ref_dict = {'UPSC':0,'Research':1,'Corporate':2}

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

nltk.download('stopwords')
nltk.download('punkt')
processed_sentences= preprocessor(list(X)).run()

for i in range(len(processed_sentences)):
  processed_sentences[i] = ' '.join(processed_sentences[i])

# vectorization

from sklearn.feature_extraction.text import TfidfVectorizer

Vectorizer = TfidfVectorizer(use_idf = True)
Vectorizer.fit(processed_sentences)

with open('Vectorizer.pickle','wb') as f:
   pickle.dump(Vectorizer,f)