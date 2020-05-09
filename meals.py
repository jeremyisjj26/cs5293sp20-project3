import pandas as pd
import spacy
import json
import re
import numpy as np
import nltk


df = pd.read_json(r'yummly.json')
df.to_csv('cuisine.csv')

df = pd.read_csv("cuisine.csv")    #read csv to pandas dataframe

df = df[['id','cuisine','ingredients']]

df.dropna(inplace=True) #Drop rows that are missing any data

stop_words = nltk.corpus.stopwords.words('english')

def normalize_document(txt):
        txt = re.sub(r'[^a-zA-Z0-9\s]', '', txt, re.I|re.A)
        txt = txt.lower()                                           #makes all text lower case
        txt = txt.strip()                                           #removes any trailing or leading white spaces
        tokens = nltk.word_tokenize(txt)                            #tokenize words
        clean_tokens = [t for t in tokens if t not in stop_words]   #get all the words in tokens where tokens is not a stop word
        return ' '.join(clean_tokens)

normalize_corpus = np.vectorize(normalize_document)                 #use numpy vectorize tool to normalize the entire document

norm_corpus = normalize_corpus(list(df['ingredients']))

print(f'{"Total Rows in data selection: "}{len(norm_corpus)}{".  First Row of text: "}{norm_corpus[0]}')

from sklearn.feature_extraction.text import TfidfVectorizer                         #import TfidVectorizer from sklearn

tf = TfidfVectorizer(ngram_range=(1,2), min_df=2)                                    #TfidVectorizer function using ngram range of 1 to 2 words or phrases.
tfidf_matrix = tf.fit_transform(norm_corpus)                                        #Calling fit_transform fuction on the new list(norm_corpus) to create an array of words to a matrix

tfidf_matrix.shape                                                              #Calling fuction to perform maching learning on new matrix

print(f'{"Observe the shape of the transformed matrix: "}{tfidf_matrix.shape}')#Observing the results of the matrix to observe size
from sklearn.metrics.pairwise import cosine_similarity

doc_sim = cosine_similarity(tfidf_matrix)           #function that runs cosine similarity function on the matrix

doc_sim_df = pd.DataFrame(doc_sim)      #creates a datafram from doc sim consine matrix

doc_sim_df.info()
doc_sim_df.head()

meals_list = df['cuisine'].values    #create a meals list into values

meal_idx = np.where(meals_list == 'mexican')[0][0]    #create meal index for american food

meal_idx      #show meal idx of mexican  cuisine

meal_similarity = doc_sim_df.iloc[meal_idx].values  #function that finds all meals similiar to the meal index of mexican

similar_meal_idx = np.argsort(-meal_similarity)[1:6]

similar_meal_idx    #displcay the top 5 meal indexs

import sys
file = open('outfile.txt','a')
sys.stdout = file

print('Top 5 Similar Cuisines based on input: ', meals_list[similar_meal_idx], meal_similarity)

file.close()
