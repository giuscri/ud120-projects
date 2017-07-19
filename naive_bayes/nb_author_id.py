#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""

from __future__ import print_function

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
start = time()
clf.fit(features_train, labels_train)
print('clf.fit() took {}ms'.format(int((time()-start)*1000)))

start = time()
labels_test_predicted = clf.predict(features_test)
print('clf.predict() took {}ms'.format(int((time()-start)*1000)))

score = 0
for actual, expected in zip(labels_test_predicted, labels_test):
    if actual == expected: score += 1./len(features_test)
print("measured accuracy for clf: {:.2f}".format(score))
#########################################################


