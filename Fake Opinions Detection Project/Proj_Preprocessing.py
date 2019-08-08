#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 15:39:36 2019

@author: Ananya
"""

import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer 
from textblob import TextBlob
from nltk import pos_tag
#from ntlk.chunk import ne_chunk
reviews=pd.read_csv("Reviews.csv")

#reviews=reviews.sample(1000)
#preprocess away
"""
########################################
function to remove punctuation from text
########################################
""" 
def remove_punctuation(df, col):
    
    df[col] = df[col].str.replace('[^\w\s]','')
    return df
    

"""
############################################
function to convert everything to lower case
############################################
""" 
def convert_to_lowercase(df, col):
    df[col] = df[col].apply(lambda x: x.lower())
    return df

"""
############################
function to remove stopwords
############################
""" 
def remove_stopwords(df, col):
    ## dictionary_of_df_words ##  
    #NUMBER OF STOP WORDS#
    """
    stopWords = set(stopwords.words('english'))
    
    wordsFiltered = []

    for w in dictionary_of_df_words :
        if w not in stopWords:
            wordsFiltered.append(w)

    print(wordsFiltered)
    """
    """
     #checking if token words of the sentence
        #are in stopWOrds
        #if they are, then remove the words
        #form the string
    """
    i=0
    stopWords = set(stopwords.words('english'))
    l=[]
    for x in df[col]:
        l_of_words=word_tokenize(x.decode('utf-8'))
        wordsFiltered = []
        for w in l_of_words:
            if w not in stopWords:
                wordsFiltered.append(w)
        
        x=" ".join(wordsFiltered)
        #print "x is",x
        l.append(x)
        i+=1;
    
    #print "length of l is", len(l[0])
    #print "length of df is", df.shape
    df[col]=l
    return df

"""
###################
Spelling Correction
###################

"""
def correct_spelling(df,col):
    l=[]
    for x in df[col]:
        x=TextBlob(x)
        x=str(x.correct())
        l.append(x)
        
    df[col]=l
    return df
        

"""
##################################
function to perform tokenization
##################################
""" 
def tokenize(df, col):
    l=[]
    
    for x in df[col]:
        word_token_list=word_tokenize(x)
        l.append( word_token_list)
        
    df[col]=l
    return df

"""
##################################
function to perform lemmatization
##################################
""" 
def lemmatize(df, col):
    lemmatizer = WordNetLemmatizer() 
    Lp=[]
    for x in df[col]:
        l=[]
        for y in x:
            y=lemmatizer.lemmatize(y,'v')
            l.append(y)
        Lp.append(l)
        
    df[col]=Lp
            
    return df

"""
#############################################
function to perform lchunking(shallow parsing)
#############################################

"""
def perform_shallow_parsing(df,col):
    l=[]
    for x in df[col]:
        tagged=ntlk.pos_tag(x)
        ent=nltk.chunk.ne_chunk(tagged)
        l.append(ent)
    
    df[col]=l  
    return df

"""
#############################################
function to solve class imbalance
#############################################

"""
def solve_class_imbalance(df,col):
    #Here we use undersampling to reach 66:44 ratio roughly
    m=df[col]
    i=0
    j=0
    for x in m:
        if x==-1:
            i=i+1
        else:
            j=j+1
    
    if (i< 3*j):
        print "we need to do something"
        #keep all -1's 
        #keep double the amount of +1's by random sampling
        #extract dataframe containing only +1's as label
        df1=df[df[col]==1]
        #extract dataframe containing only -1's
        dfm1=df[df[col]==-1]
        #randomly sample 2*i pounts
        df2=df1.sample(3*i)
        #combine both dataframes dfm1 and df2
        df=pd.concat([dfm1, df2])
        
    else:
        print "no need for doing anything"
        print "i is",i," and j=", j
        return df
            
    
    return df



print "removing punctuation"
df=remove_punctuation(reviews, "review")
print "coverting to LC"
df=convert_to_lowercase(df, "review")
print "removing stop words"
df=remove_stopwords(df, "review")
#print "correcting spelling"
#df=correct_spelling(df,"review")
print "tokenizing"
df=tokenize(df, "review")
print "lemmatizing"
df=lemmatize(df, "review")
print "solving class imbalance"
df=solve_class_imbalance(df,"label")


df.to_csv("pre_processed.csv", index=False)



