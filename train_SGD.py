#----------------------------------------------------------------------------------
# LR Classifier
# Smriti Jha
#----------------------------------------------------------------------------------
#

import csv
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier

#feature names derived from raw dataset
fieldnames = ['ACCOUNT', 'OTHER_PAGE', 'SKU', 'SEARCH_RESULTS', 'CART', 'CLASS', 'DEPARTMENT', 'HOME', 'SKUSET', 'BUY']
#training dataset
input_file = 'train.csv'
#test dataset
test_file = 'test.csv'

def readCSV(input_file):
    #read csv file into a list of lists
     new_rows = []
     with open(input_file, 'r') as test:
        test_reader = csv.reader(test)
        for row in test_reader:
            #append to temporary location in memory
            new_rows.append(row)
     return new_rows

def processCols(new_rows):
    #processes columns to get time spent per activity from cumulative time
    for index in xrange(len(new_rows)):
        for col in xrange(1, 14, 2):
            new_rows[index][col] = int(new_rows[index][col+2]) - int(new_rows[index][col])
        new_rows[index][15] = 0
    return new_rows

def createDict(new_rows, nolabel):
    data = []
    #gets the index of label in the dataset
    label_index = len(new_rows[0])-1

    #create a dictionary of features per row
    #each row is made of dict with 10 keys, including label
    for index in xrange(len(new_rows)):
        row = dict.fromkeys(fieldnames, 0)
        for col in xrange(0, 15, 2):
            row[new_rows[index][col]] = row[new_rows[index][col]] + new_rows[index][col+1]
        #calculate labels only for train set
        if(nolabel == False):
            row['BUY'] = new_rows[index][label_index]

        data.append(row)
    #returns a list of dict
    return data

def createFeature(feature):
    #convert and seperate the dictionary into an array of features and labels
    #saves the list
    dataSet = []
    for row in feature:
        #saves a row
        item = []
        #ten pre-defined fieldnames/features initialized above
        for key in fieldnames:
            item.append(row[key])
        dataSet.append(item)

    #convert to array
    dataSet = np.array(dataSet)
    return dataSet[:,:9], dataSet[:,9]

def writeTarget(target):
    print "Writing predicted target values to predicted_labels.csv..."
    with open('predicted_labels.csv', 'w') as solutions:
        sol_writer = csv.writer(solutions, delimiter=',')
        for row in target:
            #writes to solution row by row
            sol_writer.writerow(row)

def trainLR(trainingSet, label):
    #regularized logistic regression using the liblinear library, newton-cg and lbfgs solvers
    clf = SGDClassifier(loss="log", penalty="l2")
    clf.fit(trainingSet, label)
    return clf

def predictBuy(lr_clf, test_set):
    #list to save predicted labels
    predict_target = []

    for index in xrange(len(test_set)):
        #predict using a trained logistic regression model
        predict_target.append(lr_clf.predict(test_set[index]))

    return predict_target

def extract(input_file, nolabel):
    #extract from csv into memory for preprocessing
    new_rows = readCSV(input_file)
    #process times in cols and get actual time per column from cumulative time
    processed_rows = processCols(new_rows)
    #convert into a lower dimensional feature set derived from categories
    dataset = createDict(processed_rows, nolabel)
    #use the list of dictionary features to get an input matrix and label array
    feature, label = createFeature(dataset)

    return feature, label

def main():
    #call extract with input filename and nolabel boolean set to false
    feature, label = extract(input_file, False)

    print "Training a logistic regression classifier on training set..."
    #train a logistic regression classifier
    lr_clf = trainLR(feature, label)

    #discards label as we are not provided that for the test set
    test_set, label = extract(test_file, True)
    #use a logistic regression classfier to predict on test set
    predict_target = predictBuy(lr_clf, test_set)

    #print target values to file
    writeTarget(predict_target)

if __name__ == "__main__":
    main()
