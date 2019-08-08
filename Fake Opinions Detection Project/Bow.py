#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 17:26:37 2019

@author: Ananya
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    


df=pd.read_csv("pre_processed.csv")

#df=df.head(100)

    
"""
############################################
#function to TF-IDF Model of data
############################################
"""
def make_tf_idf_model(df ,my_ngram_range, col):#), tf1):
    print("making tf-idf model bow")
    """
    TRYING: TF-IDF Model 
    TFIDVectorizer
    """
    
    if my_ngram_range!=(1,1):
       # TDIDF + ngram Model
       vectorizer = TfidfVectorizer(min_df=0.05, max_df=0.9, ngram_range=my_ngram_range,use_idf=True)
    else:
       # Pure TFIDF model
       vectorizer = TfidfVectorizer(min_df=0.1, max_df=0.9, ngram_range=my_ngram_range,use_idf=True)
    
    #train_bow = vectorizer.fit_transform(df['lem_comments'])
    train_bow = vectorizer.fit_transform(df[col])
    
    # get all unique words in the corpus
    vocab = vectorizer.get_feature_names()
    print "vocab is", vocab
    #print "train_bow is", train_bow
    
    X_=train_bow.toarray()
    
  

    
    return X_

"""
#########################################################################
#function to perform feature engineering
#########################################################################
"""
def perform_feature_engineering(X_train):
    print("Performing feature engineering")
    print "before feature -mat", X_train.shape
    #X_train is my tdidf vector
    """Clustering """
    print("clustering")
    #fit the model
    k_means=KMeans(n_clusters=6).fit_transform(X_train)
    #predict 
    #sentence=["i am hateful","i love everyone","threatning","abusive","love","toxic relationship"]
    #print(kmeans.predict(tfid_vectorizer.transform(sentence)))
    print "kmeans -mat", k_means.shape
    """ Latent Dirichlet Allocation(LDA) (on Vectorizer )"""
    # Run LDA
    print("lda")
    no_topics = 500
    #lda_matrix = LatentDirichletAllocation(n_components=6, max_iter=100, random_state=0)
    lda_matrix=LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit_transform(k_means)
    #lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
    print "lda -mat", lda_matrix.shape

    """NMF (on tfidf)"""
     # Run NMF
    print("NMF")
    #init='nndsvd'
    nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5).fit_transform(lda_matrix)
    print "nmf -mat", nmf.shape

    """ Apply Truncated SVD """
    print("Truncated-SVD")
    #n_components decide how many features should be chosen (100-500) is a good range usually
    n_my_components=200
    svd_model = TruncatedSVD(n_components=n_my_components, algorithm='randomized',n_iter=10, random_state=42)
    svd_tfidf = svd_model.fit_transform(nmf)
    print "svdt -mat", svd_tfidf.shape
    return svd_tfidf



y=df['label']
del df['label']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
df, y, test_size=0.33, random_state=42)

X_train_1=make_tf_idf_model(X_train ,(1,2), "review")
X_test_1 =make_tf_idf_model(X_test ,(2,2), "review")

#perform feature engineering
X_train=perform_feature_engineering(X_train_1)
X_test=perform_feature_engineering(X_test_1)


#number of folds
n_iter_search=10

"""

##############################

"""
# Multinomial NB
from sklearn.naive_bayes import MultinomialNB

param_dist= {'alpha': [1, 1e-1, 1e-2]}
clf = MultinomialNB()
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)

rs=random_search.fit(X_train, y_train)
predicted = rs.predict(X_train)

print "Accuracy is", accuracy_score(y_train, predicted)," for NB"
#print "Accuracy is", accuracy_score(y_test, predicted)," for NB"
#clf = MultinomialNB().fit(X_test, y_test)
#predicted = clf.predict(X_test)

"""

##############################

"""


#Random Forest
from sklearn.ensemble import RandomForestClassifier

param_dist1 = {"max_depth": [3, None],
              "max_features": ['auto', 'sqrt', 'log2'],
              "min_samples_split": [2, 5, 10, 15, 100],
              "min_samples_leaf": [1, 2, 5, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
    
clf1=RandomForestClassifier()
random_search1 = RandomizedSearchCV(clf1, param_distributions=param_dist1,
                                   n_iter=n_iter_search)

rs1=random_search1.fit(X_train, y_train)
predicted1 = rs1.predict(X_train)

#clf1.fit(X_train, y_train)
#predicted1 = rs1.predict(X_train)
print "Accuracy is", accuracy_score(y_train, predicted1)," for RF"

#print "Accuracy is", accuracy_score(y_test, predicted1)," for RF"
#clf1.fit(X_test, y_test)
#predicted1 = clf1.predict(X_test)

"""

##############################

"""

#SVM
from sklearn.svm import SVC
clf2=SVC(kernel="linear", class_weight="balanced",probability=True)

param_dist2={'C':[0.001, 0.01, 0.1, 1, 10],
    'gamma' : [0.001, 0.01, 0.1, 1]
    }

random_search2 = RandomizedSearchCV(clf2, param_distributions=param_dist2,
                                   n_iter=n_iter_search)

rs2=random_search2.fit(X_train, y_train)
predicted2 = rs2.predict(X_train)

#clf2.fit(X_train,y_train)
#predicted2 = clf2.predict(X_train)
print "Accuracy is", accuracy_score(y_train, predicted2)," for SVM"

#print "Accuracy is", accuracy_score(y_test, predicted2)," for SVM"
#clf2.fit(X_test, y_test)
#predicted2 = clf2.predict(X_test)

#Multinomial Logistic Regression
from sklearn.linear_model import LogisticRegression

param_dist3 = {'penalty' : ['l2'],
    'C' : np.logspace(-4, 4, 20),
    'solver':['newton-cg','sag']
    }

clf3=LogisticRegression(multi_class="multinomial", solver="sag")
random_search3 = RandomizedSearchCV(clf3, param_distributions=param_dist3,
                                   n_iter=n_iter_search)

rs3=random_search3.fit(X_train, y_train)
predicted3 = rs3.predict(X_train)

#clf3.fit(X_train, y_train)
#predicted3 = clf3.predict(X_train)
print "Accuracy is", accuracy_score(y_train, predicted3)," for Logistic Regression"

#print "Accuracy is", accuracy_score(y_test, predicted3)," for Logistic Regression"
#clf3.fit(X_test, y_test)
#predicted3 = clf3.predict(X_test)

"""

##############################

"""

#Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier

param_dist4= {'learning_rate':[0.01,0.1,1,10],
              'max_depth':[4,6,8,10], 
              'min_samples_split':range(30,71,10)
              }
clf4=GradientBoostingClassifier()
random_search4 = RandomizedSearchCV(clf4, param_distributions=param_dist4,
                                   n_iter=n_iter_search)

rs4=random_search4.fit(X_train, y_train)
predicted4 = rs4.predict(X_train)

#clf4.fit(X_train, y_train)
#predicted4 = clf4.predict(X_train)
print "Accuracy is", accuracy_score(y_train, predicted4)," for GradientBoostingClassifier"

#print "Accuracy is", accuracy_score(y_test, predicted4)," for GradientBoostingClassifier"
#clf4.fit(X_test, y_test)
#predicted4 = clf4.predict(X_test)

"""

##############################

"""

#AdaBoost Classifier
from sklearn.ensemble import AdaBoostClassifier

param_dist5={'n_estimators':[50,75,100],
             'learning_rate':[0.001,0.01,1]}
clf5=AdaBoostClassifier()
random_search5 = RandomizedSearchCV(clf5, param_distributions=param_dist5,
                                   n_iter=n_iter_search)

rs5=random_search5.fit(X_train, y_train)
predicted5 = rs5.predict(X_train)

#clf5.fit(X_train, y_train)
#predicted5 = clf5.predict(X_train)
print "Accuracy is", accuracy_score(y_train, predicted5)," for AdaBoostClassifier"

#clf5.fit(X_test, y_test)
#predicted5 = clf5.predict(X_test)
#print "Accuracy is", accuracy_score(y_test, predicted5)," for AdaBoostClassifier"

"""

##############################

"""



