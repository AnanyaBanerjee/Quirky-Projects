#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 17:15:31 2019

@author: Ananya
"""



import pandas as pd;
import numpy as np;
from time import time
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textblob import Word
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import log_loss,confusion_matrix,classification_report,roc_curve,auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from sklearn.metrics import recall_score
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.cluster import KMeans
   
 
df=pd.read_csv("train.csv");
df_test=pd.read_csv("test.csv");
df_test_labels=pd.read_csv("test_labels.csv");

columns=list(df.columns)

print ("Columns are", columns)

unique_id=np.unique(list(df["id"]))

if(len(unique_id)==len(df["id"])):
    print ("ID is primary key")
    
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
##################################
function to perform lemmatization
##################################
""" 
def lemmatize(df, col):
     df['lem_comments'] = df[col].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
     return df


"""
################################################
##function to get the list of words in the comment
################################################
""" 

def make_my_words_dictionary(df, col):
    t=list(df[col])
    dictionary_of_words=[]
    for x in t:
        words = word_tokenize(x.decode('utf-8'))
        #words=x.split()
        dictionary_of_words.append(words)
    
    dictionary_of_words=reduce(lambda x,y: x+y,dictionary_of_words)
    #print dictionary_of_words
    return dictionary_of_words

 
    
"""
##################################
function to perform tokenization
##################################
""" 
def tokenize(df, col):
    #tokenize
    dictionary_of_words=make_my_words_dictionary(df,"comment_text")
    
    return  dictionary_of_words

   
    
"""
##################################
function to find term frequency (TF)
##################################
""" 
def find_TF(dictionary_of_words):
    
    #tf1 = (df[col]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
    tf1=pd.DataFrame()
    tf=[]
    for w in dictionary_of_words:
        z=str(dictionary_of_words.count(w))
        tf.append(z)
   
    tf1['words']=dictionary_of_words
    tf1['TF']=tf
    return tf1

    
"""
################################################
function to find inverse document frequency (IDF)
################################################
""" 
def find_IDF(df, tf1, col):
    
    for i,word in enumerate(tf1['words']):
        tf1.loc[i, 'idf'] = np.log(df.shape[0]/(len(df[df[col].str.contains(word)])))

    
    return tf1


"""
##################################
function to find term frequency-inverse document frequency (TF-IDF)
##################################
""" 
def find_TF_IDF(tf1, col):
    
    tf1['tf-idf'] = tf1['tf'] * tf1['idf']
    return tf1



    
"""
############################################
#function to perform pre-processing of data
############################################
"""
def perform_pre_processing(df, type_of_df):
    print ("pre processing",type_of_df,"dataset")
    """
    TRYING:
    
    removing punctuation from comment text

    """
    print ("removing punctuation")
    df= remove_punctuation(df,'comment_text')
    
    """
    
    TRYING:
    
    converting everything to lower case

    """
    print ("converting to lower case")
    df= convert_to_lowercase(df,'comment_text')
    
    """
    
    TRYING:
        
        removing stop words
        
    """
    print ("removing stop words")
    df=remove_stopwords(df, 'comment_text')
         
    """
    TRYING:
    (preferred over stemming)
    Lemmatization: onverts the word into its root word

    """
    print ("lemmatizing")
    df= lemmatize(df, 'comment_text')
  
    
    """
    
    Tokenization and returning list of words for df['lem_comments']
    
    it gives list of words 
    """
    if type_of_df=='train':
        print ("initializing dictionary")
        #dictionary_of_df_words=tokenize(df, 'lem_comments')
        dictionary_of_df_words=[]
        
    else:
        dictionary_of_df_words=[]
    print ("i reached ")
    """
    ############################
     
    tf1 is seperate dataset containing words;tf;idf;td-idf;
    
    ############################
    """
    """
    TRYING:
    
    Term Frequency: TF = (Number of times term T appears in the particular row) / (number of terms in that row)

    TF1 is a dataset containing words, their tf
    """

    #tf1=find_TF(df, 'lem_comments',dictionary_of_df_words)
    #tf1[tf1['words']=='you']
    
    """
    TRYING:
    
    Inverse Document Frequency:(IDF)
        a word is not of much use to us if it’s appearing in all the documents.
        IDF = log(N/n), 
        where, N is the total number of rows and n is the number of rows in which the word was present.

     TF1 is a dataset containing words, their tf, idf
    """
   
    #tf1=find_IDF(df, tf1, 'lem_comments')
    
    
    
    """
    TRYING:
    
    Term Frequency – Inverse Document Frequency (TF-IDF)
     
    tf1 is a dataset containing: words, tf, idf, tf-idf
    """
    #tf1=find_TF_IDF(tf1, 'lem_comments')
    
    #return df, tf1,dictionary_of_df_words
    return df, dictionary_of_df_words
   
    
"""
#########################################################################
#function to check document similarity to engineer task specific features
#########################################################################
"""
def document_similarity(X_train):
    """### COSINE SIMILARITY ###"""
    #Cosine similarity basically gives us a
    #metric representing the cosine of the angle 
    #between the feature vector representations 
    #of two text documents. 
    #Lower the angle between the documents,
    #the closer and more similar
    similarity_matrix = cosine_similarity(X_train)
    similarity_df = pd.DataFrame(similarity_matrix)
    
    
    return similarity_df



"""
#########################################################################
#function to perform feature engineering
#########################################################################
"""
def perform_feature_engineering(X_train):
    print("Performing feature engineering")
    #X_train is my tdidf vector
    """Clustering """
    print("clustering")
    #fit the model
    k_means=KMeans(n_clusters=6).fit_transform(X_train)
    #predict 
    #sentence=["i am hateful","i love everyone","threatning","abusive","love","toxic relationship"]
    #print(kmeans.predict(tfid_vectorizer.transform(sentence)))
  
    """ Latent Dirichlet Allocation(LDA) (on Vectorizer )"""
    # Run LDA
    print("lda")
    no_topics = 500
    #lda_matrix = LatentDirichletAllocation(n_components=6, max_iter=100, random_state=0)
    lda_matrix=LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit_transform(k_means)
    #lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)

    """NMF (on tfidf)"""
     # Run NMF
    print("NMF")
    #init='nndsvd'
    nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5).fit_transform(lda_matrix)


    """ Apply Truncated SVD """
    print("Truncated-SVD")
    #n_components decide how many features should be chosen (100-500) is a good range usually
    n_my_components=200
    svd_model = TruncatedSVD(n_components=n_my_components, algorithm='randomized',n_iter=10, random_state=42)
    svd_tfidf = svd_model.fit_transform(nmf)
    
    return svd_tfidf

    

"""
############################################
#function to bag of words of data
############################################
"""
def make_bag_of_words(df,dictionary_of_df_words):#), tf1):
    print("making bow")
    """
    TRYING:
    
    Bag of Words:representation of text which describes the presence of words within the text data.
     
     two similar text fields will contain similar kind of words, and will therefore have a similar bag of words. Further, 
     that from the text alone we can learn something about the meaning of the document.
        
    """
    bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1,1),analyzer = "word")

    train_bow = bow.fit_transform(df['lem_comments'])
    
    test_bow=bow.fit_transform(df_test['comment_text'])
    
    #print(train_bow.toarray())
    
    #print(len(train_bow.toarray()))
    #gives names of features
    #print "The feature names are"
    #print(bow.get_feature_names())
    
    X_train=train_bow.toarray()
    X_test=test_bow.toarray()
    
    # get all unique words in the corpus
    vocab = bow.get_feature_names()
    # show document feature vectors
    BOW=pd.DataFrame(X_train)#, columns=vocab)
    BOW.columns=vocab
    
    #Feature Engineering both training and test
    X_train=perform_feature_engineering(X_train)
    X_test=perform_feature_engineering(X_test)
    
    """
    TFIDVectorizer
    
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(df['lem_comments'])
    print(vectorizer.get_feature_names())
    
    test_bow=vectorizer.fit_transform(df_test['comment_text'])
    X_test=test_bow.toarray()
    """
    
    ##################
    return X_train, X_test, BOW


"""
############################################
#function to N Gram -bag of words of data
############################################
"""
def make_bag_of_words_N_grams(df,dictionary_of_df_words):#), tf1):
    print("making n-gram bow (2,2)")
    """
    TRYING: N-Gram BOW
    
    Bag of Words:representation of text which describes the presence of words within the text data.
     
     two similar text fields will contain similar kind of words, and will therefore have a similar bag of words. Further, 
     that from the text alone we can learn something about the meaning of the document.
        
    """
    bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(2,2),analyzer = "word")

    train_bow = bow.fit_transform(df['lem_comments'])
    
    test_bow=bow.fit_transform(df_test['comment_text'])
    
    #print(train_bow.toarray())
    
    #print(len(train_bow.toarray()))
    #gives names of features
    #print "The feature names are"
    #print(bow.get_feature_names())
    
    X_train=train_bow.toarray()
    X_test=test_bow.toarray()
    
    # get all unique words in the corpus
    vocab = bow.get_feature_names()
    # show document feature vectors
    N_BOW=pd.DataFrame(X_train)#, columns=vocab)
    N_BOW.columns=vocab
    
    #Feature Engineering both training and test
    X_train=perform_feature_engineering(X_train)
    X_test=perform_feature_engineering(X_test)
   
   
    return X_train, X_test, N_BOW
    
"""
############################################
#function to TF-IDF Model of data
############################################
"""
def make_tf_idf_model(df,df_test, dictionary_of_df_words,my_ngram_range):#), tf1):
    print("making tf-idf model bow")
    """
    TRYING: TF-IDF Model 
    TFIDVectorizer
    """
    """
    import gensim.downloader as api
    from gensim.models import TfidfModel
    from gensim.corpora import Dictionary

    dataset = api.load("text8")
    dct = Dictionary(dataset)  # fit dictionary
    corpus = [dct.doc2bow(line) for line in dataset]  # convert corpus to BoW format

    model = TfidfModel(corpus)  # fit model
    vector = model[corpus[0]]  # apply model to the first corpus document
    """
    if my_ngram_range!=(1,1):
       # TDIDF + ngram Model
       vectorizer = TfidfVectorizer(min_df=0.05, max_df=0.9, ngram_range=my_ngram_range,use_idf=True)
    else:
       # Pure TFIDF model
       vectorizer = TfidfVectorizer(min_df=0.1, max_df=0.9, ngram_range=my_ngram_range,use_idf=True)
    
    train_bow = vectorizer.fit_transform(df['lem_comments'])
    # get all unique words in the corpus
    vocab = vectorizer.get_feature_names()
    
    #test_bow=vectorizer.fit_transform(df_test['comment_text'])
    test_bow=vectorizer.transform(df_test['comment_text'])
    
    #print(vectorizer.get_feature_names())
    X_train=train_bow.toarray()
    X_test=test_bow.toarray()
    
    # show document feature vectors
    TFID_Model=pd.DataFrame(np.round(X_train, 2))#, columns=vocab)
    TFID_Model.columns=vocab
    
    #Feature Engineering both training and test
    X_train=perform_feature_engineering(X_train)
    X_test=perform_feature_engineering(X_test)
   
    
    return X_train, X_test, TFID_Model

"""
#######################################################
#function to evaluate performance of model on test data
#######################################################
"""
def eval_test_label(df_test_labels, y_test_pred, label):
    print("Evaluating test results")
    #now our test dataset has -1 to indictae that comment has not been used to score the label
    #make a col of label_pred in df_test_labels
    df_test_labels[label+'_pred']=y_test_pred
    #filter the dataset where labels!=-1
    df1=df_test_labels[df_test_labels[label]!=-1]
    #get actual and test values
    test_actual=df1[label]
    test_pred=df1[label+'_pred']
    #find accuracy on test set now
    print("accuracy on test set for label",label,"is", accuracy_score(test_actual,test_pred ))
    acc=accuracy_score(test_actual,test_pred )
    #find recall on test set
    recall=recall_score(test_actual,test_pred )
    #print("recall for test set for label=",label,"is", recall)
    return acc,recall

        
"""
############################################
#function to make a model and evaluate 
############################################
"""
def make_model(df, df_test, X_train, X_test, df_test_labels):
    print ("model making")
    #dictionary to store train accuracy
    train_acc=dict()
    #dictionary to store testvalues for each label
    test_val=dict()
    #form model
    """### Logistic Regression ###"""
    model=LogisticRegression(C=2.0)
    param_dist = {'penalty' : ['l1', 'l2'],
    'C' : np.logspace(-4, 4, 20),
    'solver' : ['liblinear']}
    n_iter_search = 5
    """#### Random Forest With RandomizedSearchCV #####"""
    """
    # specify parameters and distributions to sample from
    param_dist = {"max_depth": [3, None],
              "max_features": ['auto', 'sqrt', 'log2'],
              "min_samples_split": [2, 5, 10, 15, 100],
              "min_samples_leaf": [1, 2, 5, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
    model=RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    
    #sp_randint(1, 11)

    # run randomized search
    n_iter_search = 20
    """
    random_search = RandomizedSearchCV(model, param_distributions=param_dist,
                                   n_iter=n_iter_search)

    #declare target labels list
    target_label=['toxic','severe_toxic','obscene','threat','insult','identity_hate']
    for label in target_label:
        print ("For "+label)
        #get corresponding label values for train 
        y_train=df[label]
        #fit the model
        start = time()
        rs=random_search.fit(X_train, y_train)
        print("RandomizedSearchCV took %.2f seconds for %d candidates"
        " parameter settings." % ((time() - start), n_iter_search))
        #report(random_search.grid_scores_)
        #finding the predicted prob for X_test
        #y_test_pred_prob=random_search.predict_proba(X_test)[:,1]
        y_test_pred_prob=rs.predict_proba(X_test)[:,1]
        
        #model.fit(X_train, y_train)
        #find the training accuracy
        #y_pred_train=random_search.predict(X_train)
        y_pred_train=rs.predict(X_train)
        
        print ("Training accuracy for label=",label,"is",accuracy_score(y_train,y_pred_train))
        
        #finding the predicted prob for X_test
        #y_test_pred_prob=model.predict_proba(X_test)[:,1]
        #y_test_pred_prob=random_search.predict_proba(X_test)[:,1]
        
        #get corresponding label values for train 
        #y_test_actual=df_test_labels[label]
        #y_test_pred=random_search.predict(X_test)
        y_test_pred=rs.predict(X_test)
        
        t_acc, recall=eval_test_label(df_test_labels, y_test_pred, label)
        #fill test dictionary
        test_val[label]=[y_test_pred_prob,t_acc,y_test_pred, recall]
        #confusion matrix for training data
        col = label
        print   ("Column:",col)
        #pred =  lr.predict(X)
        print('\nConfusion matrix\n',confusion_matrix(y_train,y_pred_train)) 
        tn, fp, fn, tp=confusion_matrix(y_train,y_pred_train).ravel()
        print(classification_report(y_train,y_pred_train))
        train_acc[label]=[y_train, y_pred_train,accuracy_score(y_train,y_pred_train),classification_report(y_train,y_pred_train),tn, fp, fn, tp]
        
    return train_acc, test_val



"""
###########
MAIN
###########
"""     
  
"""###Pre processing training data###"""
#df, tf1,dictionary_of_df_words=perform_pre_processing(df)
df,dictionary_of_df_words=perform_pre_processing(df,'train')

"""###Pre processing test data###"""
#df, tf1,dictionary_of_df_words=perform_pre_processing(df)
df_test,dictionary_of_df_test_words=perform_pre_processing(df_test,'test')

"""###Making Bag of words###"""
#X_train, X_test, BOW=make_bag_of_words(df,dictionary_of_df_words)

"""###Making N Gram BOW###"""
#X_train, X_test, N_BOW=make_bag_of_words_N_grams(df,dictionary_of_df_words)

"""###Making TF-IDF Model###"""
#ngram_range=(1,1)
#X_train, X_test, TFID_Model=make_tf_idf_model(df,df_test, dictionary_of_df_words,ngram_range)

"""###Making TF-IDF Model(2,2)###"""
ngram_range=(2,2)
X_train, X_test, TFID_Model=make_tf_idf_model(df,df_test, dictionary_of_df_words,ngram_range)

"""###Making model and finding accuracy for model"""
train_acc, test_val=make_model(df, df_test, X_train, X_test, df_test_labels)
    
"""Testing Miss rate for Training Set"""
#train_acc[label]=[y_pred_train,accuracy_score(y_train,y_pred_train),classification_report(y_train,y_pred_train),tn, fp, fn, tp]
for label in train_acc.keys():
    op=train_acc[label]
    TN=op[4]
    FP=op[5]
    FN=op[6]
    TP=op[7]
    miss_rate=float(FN)/float(FN+TP)
    #print("The miss-rate for label=",label,"is",miss_rate)
    orig=op[0] #original values
    pred=op[1] #predicted values
    #print("Recall for label:",label,"is",recall_score(orig, pred))

""" Saving the test data that we predicted in submission.csv """
col_final=['toxic','severe_toxic','obscene','threat','insult','identity_hate']
final_df = pd.DataFrame(columns=col_final)
#dictionary test_val[label]=[y_test_pred_prob,t_acc,y_test_pred, recall]
for label in test_val.keys():
    li=test_val[label]
    label_col=li[0]
    final_df[label]=label_col
#add id column from test dataset    
submit = pd.concat([df_test['id'],final_df],axis=1)
#submit.to_csv('final_submission.csv',index=False)
#submit.head()