#
# -2: you have a bug in your code such that none of the parameters are
# actually changed in the classifier.
#
# Also, you need to normalize data before using kNN.
#
# Jia Feng Yu
# 1830021
# Professor Robert Vincent
# May 31 2020
# Programming Techniques & Applications
# Assignment 3
from extra_trees import extra_trees # suggestion!
from classifier import data_item, normalize_dataset
from knnclassifier import LabeledPoint, knnclassifier
from random import shuffle

fp = open('spambase.data')
dataset = []
for line in fp:
    fields = line.split(',')
    data = [float(x) for x in fields[:-1]]
    label = int(fields[-1])
    dataset.append(data_item(label, data))

print("Read {} items.".format(len(dataset)))
print("{} features per item.".format(len(dataset[0].data)))

# Add your code here...
n = len(dataset) #total length of the database
def confusion_matrix(classifier_used, **args): #**args for specific arguments of the classifier
    classifier = classifier_used
    all_tpr = [] #list to append all values of tpr to calculate avg later
    all_fpr = [] #same but for fpr
    for i in range(5):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        shuffle(dataset)
        training_set = dataset[:int(0.8*n)+1] #+1 such that the last item is included since the closing index is exclusive
        test_set = dataset[int(0.8*n)+1:]#+1 such that there isn't a duplicate of the 3681th item
        classifier.train(training_set)
        for i in test_set: #for every object in the test set
            if classifier.predict(i.data) == 1:
                if i.label == 1:
                    tp += 1
                else:
                    fp += 1
            elif classifier.predict(i.data) == 0:
                if i.label == 0:
                    tn += 1
                else:
                    fn += 1
        tpr = tp/(tp + fn)
        fpr = fp/(fp + tn)
        all_tpr.append(tpr) #adds tpr to the list
        all_fpr.append(fpr) #adds fpr to the list
        print("Confusion Matrix: 0     1")
        print("                0",tn, " ", fn)
        print("                1",fp, " ", tp) #confusion matrix
    sum = 0
    for j in all_tpr:
         sum += j #adds all tprs
    avg_tpr = sum/len(all_tpr) #calculates avg tpr
    sum1 = 0
    for k in all_fpr:
        sum1 += k #adds all fprs
    avg_fpr = sum1/len(all_tpr) #calculates avg fpr
    print("TPR =",avg_tpr,", FPR = ",avg_fpr)
confusion_matrix(extra_trees())#base case
confusion_matrix(extra_trees(),M=20) #more trees
confusion_matrix(extra_trees(),K=20) #more tests to evaluate
confusion_matrix(extra_trees(),Nmin=5) #higher minimum split size
confusion_matrix(knnclassifier())#base case
confusion_matrix(knnclassifier(),K=3)
confusion_matrix(knnclassifier(),K=8)
confusion_matrix(knnclassifier(),K=12)
