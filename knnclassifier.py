'''A simple implementation of k nearest neighbor classificiation that 
   attempts to use a K-D tree to organize the training points.
'''
from classifier import *
from kdtree import *
from Geometry import Point

class LabeledPoint(Point):
    '''This extension of the Point class just adds a label we
    can use for the training data.'''
    def __init__(self, iterable, label):
        super().__init__(iterable)
        self.label = label

class knnclassifier(classifier):
    '''A simple kNN classifier based on a K-D tree.'''
    def __init__(self, K = 1):
        self.k = K

    def train(self, train_data):
        '''Train the classifier with a list of data_item objects.'''
        self.data = kdtree(len(train_data[0].data))
        for item in train_data:
            self.data.insert(LabeledPoint(item.data, item.label))

    def predict(self, data_point):
        '''Predict the class of 'data_point' by majority vote of the 'k' 
        closest points in the 'training' dataset.'''
        
        nearest = self.data.k_nearest(data_point, self.k)
        
        # now we have all items sorted by increasing distance.
        
        counts = {}
        for pt in nearest[:self.k]:
            counts[pt.label] = counts.get(pt.label, 0) + 1
        return max(counts.keys(), key=lambda key: counts[key])
    

if __name__ == "__main__":
    from datasets import read_datasets

    datasets = read_datasets()
    for name in sorted(datasets):
        dataset = datasets[name]
        print(name)
        # Create a normalized dataset.
        #
        norm_dataset = normalize_dataset(dataset)

        print("k_nearest:")
        for k in range(1, 14, 2):
            pct = evaluate(dataset, knnclassifier, 4, K=k)
            print('K {}: {:.2%}'.format(k, pct))
        print("k_nearest, normalized:")
        for k in range(1, 14, 2):
            pct = evaluate(norm_dataset, knnclassifier, 4, K=k)
            print('K {}: {:.2%}'.format(k, pct))
