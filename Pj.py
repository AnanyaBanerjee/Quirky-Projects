#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 15:05:54 2019

@author: Ananya
"""

import pandas as pd
import numpy as np

#Reading meta data
col=["user_id", "prod_id","rating","label","date"]
meta_data = pd.read_table('metadata.txt', delim_whitespace=True, names=col)

print(meta_data.shape)


#Reading reviewContent data
col1=["user_id","prod_id","date","review"]
review_data = pd.read_table('reviewContent.txt', delimiter="\t", names=col1)

print(review_data.shape)

#joining two dataframe to form a combined dataframe
#cf=pd.merge(meta_data,review_data, on=['user_id','prod_id','date'])

cf=pd.merge(meta_data,review_data[['user_id','review','prod_id','date']], on=['user_id','prod_id','date'])

cf.to_csv("Reviews.csv",index=False)
