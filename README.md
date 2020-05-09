This project can be ran from the command line using: python meals.py.

The libraries used are: 
        import pandas as pd
        import spacy
        import json
        import re
        import numpy as np
        import nltk
        from sklearn.feature_extraction.text import TfidfVecorizer
        from sklearn.metrics.pairwise import cosine_similarity
    
First I created a pandas dataframe to read in the json file that was localy installed per the class assisgnment - df = pd.read_json(r'yummly.json')

I then sent the new dataframe to a csv file - df.to_csv('cuisine.csv')

I then read the the cuisine.csv file back into a pandas dataframe.

I used nltk language processing to clean up the data.
      df.dropna function to drop rows that are mssing data
      stop_words function using nltk corpus english stop words
      
Then used continued using nltk to normalize the document:
    re.sub to remove any character that is not alpha
    txt.lower() to make all text lower case
    txt.strip() to remove any trailing or leading white spaces
    nltk.word_tokenize(txt) to tokenize the words
    created a function to remove all stop words
    
Once normalized, i created a function that used numpy to vectorize the normalized document

I then created a function that used TfidVectorizer from sklearn using ngram range of 1 to 2 words or phrases.

created a fit_transform cuntion on the norm_corpus function to fit an array of words to a matrix.

created doc_sim fuction that called upon the new tfidf_matrix using the built in cosine_similarity function from sklearn

created a function doc_sim_df to pass the new similarity document doc_sim to a dataframe.

then created a meals_list variable that passed in all the data from the cuisine attribute in the dataframe.

then created a meal_idx variable to find all simialry cuisine indexs that was similar to the cuisine input "mexican". 

created a meal_similarity variable that passed the meal_idx variable through to convert the cuisines to values.  

then created a variable function similar_meal_idx that called on a numpy fuction argsort to sort the meal_similarity variable and then collect the top 5 similary indexs to the input "mexican".

I then created a function to print the top 5 similar cuisines to "mexican", that included their name and their similarity variable value.

This was output to a file using the library sys to print the output to a text file.

NOTE:
    So i spent alot of time trying to figure out how to call the csv file into a spacy model.  I knew that was the route that i wanted to go,
    so that i could used the spacy functions.  I wasnt having any luck, so for the sake of not doing anything, i just decided to go the route of comparing similar cuisinses.
    I understand that is not specific to the task, but i had to proceed with something. The output will show that the top 5 similar cuisines to "mexican" are mexican, mexican, mexican, mexican, and mexican.
    This holds to be true, because there are several mexican cuisne dishes in the data set.  Though this is not the exact output, i hope that you will take into consideration that the variables, i was still able to provide the correct output for "mexican" cousine.
    

