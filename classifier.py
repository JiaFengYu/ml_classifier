class data_item(object):
    '''Class to represent a generic labeled
    data item.'''
    def __init__(self, label, data):
        self.label = label
        self.data = data

def normalize_dataset(dataset):
    '''Normalize all features to lie within the interval [0, 1].'''
    mins = list(dataset[0].data)
    maxs = list(dataset[0].data)
    m = len(mins)
    for i in range(m):
        features = [item.data[i] for item in dataset]
        mins[i] = min(features)
        maxs[i] = max(features)

    result = []
    for item in dataset:
        norm_data = [(v - n) / (x - n) for v, n, x in zip(item.data, mins, maxs)]
        result.append(data_item(item.label, norm_data))
    return result
    
def distance(d1, d2):
    '''n-dimensional distance between 'd1' and 'd2'.'''
    assert len(d1) == len(d2)
    r = 0
    for x, y in zip(d1, d2):
        d = x - y
        r += d * d
    return r

'''Very many machine-learning problems use the argmin and argmax
functions. These functions are widely used in optimization problems
of all sorts. The idea is simple - return the value of x for which the
function y = f(x) is a maximum. In many ML problems the values of x are
in a small discrete set, so it's easy to perform the computation as 
done in the following functions.'''

def argmax(lst):
    '''Return the index of the largest value in 'lst'.'''
    return max(range(len(lst)), key=lambda i: lst[i])
            
def argmin(lst):
    '''Return the index of the smallest value in 'lst'.'''
    return min(range(len(lst)), key=lambda x: lst[x])

class classifier(object):
    '''Generic interface for a classifier.'''
    def __init__(self):
        '''Initialize the generic classifier.'''
        pass

    def train(self, train_data):
        '''Train a classifier using the 'train_data', which is a list of 
        data_item objects.'''
        pass

    def predict(self, data_point):
        '''Given a new data point, return the most likely label
        for that value.'''
        pass

def evaluate(dataset, cls, n_folds = 0, **kwargs):
    '''Evaluate the classifier on the dataset using 'n_folds' of
    cross-validation. If n_folds is equal to zero, the code performs
    leave-one-out cross-validation.'''
    if n_folds == 0:
        n_folds = len(dataset)
    n_features = len(dataset[0].data)
    test_size = round(len(dataset) / n_folds)
    index = 0
    n_correct = 0           # count correct predictions.
    n_tested = 0
    for fold in range(n_folds):
        train_data = dataset[:index] + dataset[index + test_size:]
        test_data = dataset[index:index + test_size]
        p = cls(**kwargs)
        p.train(train_data)
        for item in test_data:
            if p.predict(item.data) == item.label:
                n_correct += 1
            n_tested += 1
        index += test_size
    return n_correct / n_tested

