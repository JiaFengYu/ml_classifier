# Simple code for loading some standard data sets from the UCI Machine
# Learning repository.
#
from classifier import data_item

def read_wine_dataset():
    '''Return a list of data_item object representing the UCI Wine data.'''
    dataset = []
    fp = open('wine.txt')
    for line in fp:
        fields = line.split(',')
        data = [float(v) for v in fields[1:]]
        label = int(fields[0]) - 1
        dataset.append(data_item(label, data))
    fp.close()
    return dataset

def read_iris_dataset():
    '''Return a list of data_item object representing the UCI Iris data.'''
    dataset = []
    fp = open('iris.txt')
    for line in fp:
        if not line.startswith('#'):
            fields = line.split()
            data = [float(v) for v in fields[:-1]]
            if fields[-1] == "Iris-setosa":
                label = 0
            elif fields[-1] == "Iris-versicolor":
                label = 1
            elif fields[-1] == "Iris-virginica":
                label = 2
            else:
                raise ValueError("Illegal class name: " + fields[-1])
            dataset.append(data_item(label, data))
    fp.close()
    return dataset

def read_seeds_dataset():
    '''Return a list of data_item object representing the UCI Seeds data.'''
    fp = open('seeds.txt')
    dataset = []
    for line in fp:
        fields = line.split()
        data = [float(v) for v in fields[:-1]]
        label = int(fields[-1]) - 1
        dataset.append(data_item(label, data))
    return dataset

def read_parkinsons_dataset():
    '''Return a list of data_item object representing the UCI Parkinson's
    data.'''
    fp = open('parkinsons.data')
    dataset = []
    header = fp.readline()
    for line in fp:
        fields = line.split(',')
        label = int(fields[17])
        data = [float(x) for x in (fields[1:17] + fields[17+1:])]
        dataset.append(data_item(label, data))
    fp.close()
    return dataset

def read_datasets():
    '''Return all four of the datasets we use into a single dictionary.'''
    datasets = {}
    datasets['Wine'] = read_wine_dataset()
    datasets['Iris'] = read_iris_dataset()
    datasets['Seeds'] = read_seeds_dataset()
    datasets['Parkinsons'] = read_parkinsons_dataset()
    return datasets
