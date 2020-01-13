__author__ = 'Mina37'

import numpy
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import os
import csv
import numpy as np

################################################
##################### SVM ######################
################################################

x = []
y = []
with open('../Rsrc/train.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count = 1
        else:
            if(row[4] == 'male'):
                if(row[5] == ''):
                    x.append([0,25,row[6],row[7]])
                else:
                    x.append([0,row[5],row[6],row[7]])
            else:
                if(row[5] == ''):
                    x.append([1,25,row[6],row[7]])
                else:
                    x.append([1,row[5],row[6],row[7]])
            y.append(row[1])

clf = svm.SVC()
clf.fit(x, y)

predicted = []

with open('../Rsrc/test.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count = 1
        else:
            if(row[3]== 'male'):
                if(row[4] == ''):
                    predicted.append([0,25,row[5],row[6]])
                else:
                    predicted.append([0,row[4],row[5],row[6]])
            else:
                if(row[4] == ''):
                    predicted.append([1,25,row[5],row[6]])
                else:
                    predicted.append([1,row[4],row[5],row[6]])
pred = clf.predict(predicted)
count = 0

with open('../Rsrc/gender_submission.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count = 1
        else:
            if(pred[line_count - 1] == row[1]):
                count +=1
            line_count +=1

print('SVM accuracy = ')
print(count/len(pred))

################################################
#################### Bayes #####################
################################################

mnb = GaussianNB()
mnb.fit(np.array(x).astype(np.float),np.array(y).astype(np.float))
pred = mnb.predict(np.array(predicted).astype(np.float))
count = 0
with open('../Rsrc/gender_submission.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count = 1
        else:
            if(float(pred[line_count - 1]) == float(row[1])):
                count +=1
            line_count +=1

print('Bayes accuracy = ')
print(count/len(pred))

################################################
############ Multi Layer Preceptron ############
################################################

MLP = MLPClassifier(hidden_layer_sizes = 2, batch_size = 5, random_state = 5)
MLP.fit(np.array(x).astype(np.float),np.array(y).astype(np.float))
pred = MLP.predict(np.array(predicted).astype(np.float))

count = 0
with open('../Rsrc/gender_submission.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count = 1
        else:
            if(float(pred[line_count - 1]) == float(row[1])):
                count +=1
            line_count +=1

print('Multi Layer Preceptron accuracy, Hidden layer size 2 = ')
print(count/len(pred))

MLP = MLPClassifier(hidden_layer_sizes = 6, batch_size = 5, random_state = 5)
MLP.fit(np.array(x).astype(np.float),np.array(y).astype(np.float))
pred = MLP.predict(np.array(predicted).astype(np.float))

count = 0
with open('../Rsrc/gender_submission.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count = 1
        else:
            if(float(pred[line_count - 1]) == float(row[1])):
                count +=1
            line_count +=1

print('Multi Layer Preceptron accuracy, Hidden layer size 6 = ')
print(count/len(pred))


MLP = MLPClassifier(hidden_layer_sizes = 8, batch_size = 5, random_state = 5)
MLP.fit(np.array(x).astype(np.float),np.array(y).astype(np.float))
pred = MLP.predict(np.array(predicted).astype(np.float))

count = 0
with open('../Rsrc/gender_submission.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count = 1
        else:
            if(float(pred[line_count - 1]) == float(row[1])):
                count +=1
            line_count +=1

print('Multi Layer Preceptron accuracy, Hidden layer size 8 = ')
print(count/len(pred))

MLP = MLPClassifier(hidden_layer_sizes = 15, batch_size = 5, random_state = 5)
MLP.fit(np.array(x).astype(np.float),np.array(y).astype(np.float))
pred = MLP.predict(np.array(predicted).astype(np.float))

count = 0
with open('../Rsrc/gender_submission.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count = 1
        else:
            if(float(pred[line_count - 1]) == float(row[1])):
                count +=1
            line_count +=1

print('Multi Layer Preceptron accuracy, Hidden layer size 15 = ')
print(count/len(pred))

MLP = MLPClassifier(hidden_layer_sizes = 100, batch_size = 5, random_state = 5)
MLP.fit(np.array(x).astype(np.float),np.array(y).astype(np.float))
pred = MLP.predict(np.array(predicted).astype(np.float))

count = 0
with open('../Rsrc/gender_submission.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count = 1
        else:
            if(float(pred[line_count - 1]) == float(row[1])):
                count +=1
            line_count +=1

print('Multi Layer Preceptron accuracy, Hidden layer size 100 = ')
print(count/len(pred))



print('Hello World!')
print( os.getcwd())
