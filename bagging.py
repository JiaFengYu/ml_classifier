'''Bagging is a general 'ensemble method' in machine learning. To avoid 
overfitting the training data, we construct several classifiers using 
different distributions of training examples.'''

from classifier import classifier
from decision_tree import greedy_decision_tree
from random import choice

def sample_with_replacement(lst):
    '''Return a resampled data set based on 'lst'.'''
    return [choice(lst) for _ in range(len(lst))]
    
class bagging_trees(classifier):
    '''Implement a simple version of bagging trees, in a random forest
    classifier.'''
    def __init__(self, M = 21):
        '''Initialize the empty forest.'''
        self.forest = []
        self.M = M

    def predict(self, data_point):
        '''Predict the class of the 'data_point' by majority vote.'''
        c = sum(dt.predict(data_point) > 0 for dt in self.forest)
        return c > (self.M - c)

    def train(self, training_data):
        '''Train a forest using the bagging trees algorithm.'''
        for i in range(self.M):
            dt = greedy_decision_tree()
            dt.train(sample_with_replacement(training_data))
            self.forest.append(dt)
            
if __name__ == "__main__":
    # Basic testing code.
    from datasets import read_parkinsons_dataset
    from classifier import evaluate
    from random import shuffle
    dataset = read_parkinsons_dataset()
    shuffle(dataset)
    pct = evaluate(dataset, bagging_trees, 4, M = 49)
    print('Bagging: {:.2%}'.format(pct))
