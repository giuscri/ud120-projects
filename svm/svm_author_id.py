#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
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
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

n = len(features_train)
#features_train = features_train[:int(n/100)]
#labels_train = labels_train[:int(n/100)]

C = 10000.
clf = SVC(kernel='rbf', C=C)

start = time()
clf.fit(features_train, labels_train)
print('[C={}] SVC took {} ms to train'.format(C, int((time()-start)*1000)))

start = time()
labels_test_predicted = clf.predict(features_test)
print('[C={}] SVC took {} ms to make test predictions'.format(C, int((time()-start)*1000)))

score = accuracy_score(labels_test, labels_test_predicted)
print('[C={}] score for SVC is {:.2f}'.format(C, score))

p0, p1, p2 = labels_test_predicted[10, 26, 50]
print('{}, {}, {}', p0, p1, p2)
#for C in (10., 1000., 100., 10000.):
#    clf = SVC(kernel='rbf', C=C)
#    clf.fit(features_train, labels_train)
#    print('[C={}] SVC took {} ms to train'.format(C, int((time()-start)*1000)))
#
#    start = time()
#    labels_test_predicted = clf.predict(features_test)
#    print('[C={}] SVC took {} ms to make test predictions'.format(C, int((time()-start)*1000)))
#
#    score = accuracy_score(labels_test, labels_test_predicted)
#    print('[C={}] score for SVC is {:.2f}'.format(C, score))
#########################################################


