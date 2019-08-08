The reviewContent file contains the review text and metadata file contains meta information in the following order
 
user_id 
prod_id 
rating 
label 
date

For feature extraction you need both files. For running the LBP you need the reviewGraph (user_id(1 .. N), prod_id(1 ... M), rating) file. The other two files are just mapping of original user_id and product_id to a unique number for reviewGraph representation. Based on this structure you might need to change some lines of code in initialization.


Please cite: (if you are using this work)

@inproceedings{DBLP:conf/sigkdd/Akoglu15,
author = {Shebuti Rayana and Leman Akoglu},
title = {Collective Opinion Spam Detection: Bridging Review Networks and metadata},
booktitle = {Proceeding of the 21st ACM SI/Users/Mammu/Desktop/BIG DATA project/reviewContent.txtGKDD international conference
on Knowledge discovery and data mining, {KDD’15}},
year = {2015},
}