'''Basic code and algorithms for decision trees.

A few general notes:

1. The code here only model numerical features (integer or
float). Categorical, boolean, and string features are also possible
with decision trees, with a small amount of extra work.

2. This code only supports two-class (yes/no) classification.

3. Decision trees are almost always implemented as binary trees, where
an internal node represents an individual yes/no decision, and leaf nodes
yield a class label.
'''

from classifier import classifier
from math import log2


class dt_node(object):
    '''Decision tree node.'''
    def __init__(self):
        '''Initialize a decision tree node.'''
        self.label = None       # Class label (leaf node only).
        self.index = None       # Feature index.
        self.value = None       # Feature split value.
        self.left = None        # Left subtree.
        self.right = None       # Right subtree.
        
class decision_tree(classifier):
    '''Interface for a two-class decision tree classifier.'''
    def __init__(self):
        '''Initialize a decision tree classifier.'''
        self.root = None
        
    def train(self, train_data):
        '''There are many training algorithms for decision
        trees, the default here is just a stub.'''
        pass

    def predict(self, data_point):
        '''Predict the class label of a new data point.
        A nice thing about most decision trees is that prediction is
        very fast and generally follows the same plan.'''
        x = self.root
        if x == None:
            return None         # tree is empty
        while x.left:
            if data_point[x.index] < x.value:
                x = x.left
            else:
                x = x.right
        return x.label

def split(tset, index, value):
    '''Split a training set into left and right parts.'''
    lset = []
    rset = []
    for item in tset:
        if item.data[index] >= value:
            rset.append(item)
        else:
            lset.append(item)
    return lset, rset

def score_test(tset, index, value):
    '''
    Calculate the score for this test in a classification problem.
    This score measures the 'information gain' associated with the 
    split resulting from the test.
    This is taken directly from the Appendix in Geurts et al 2006,
    which is taken in turn from Quinlan 1986 and other sources.
    '''
    def log2e(x):
        '''Log base 2 for entropy calculation.'''
        return log2(x) if x != 0.0 else 0.0

    def entropy(p):
        '''Compute the Shannon entropy of a distribution.'''
        return -sum(x * log2e(x) for x in p)

    def H_C(tset): 
        '''Calculate the entropy of the class labels of the
        training set.'''
        tlen = len(tset)
        npos = sum(x.label > 0 for x in tset) # number of positive labels.
        return entropy([npos / tlen, (tlen - npos) / tlen])

    def H_S(lset, rset):
        '''Calculates the 'split entropy' of this particular split 
        of the training set.'''
        llen = len(lset)
        rlen = len(rset)
        tlen = llen + rlen
        return entropy([llen / tlen, rlen / tlen])
    
    def H_CS(lset, rset):
        '''Calculates the average conditional entropy of the labels of
        this split of training set. This is used to calculate the 
        information gain of the split outcome and classification.'''
        llen = len(lset)
        rlen = len(rset)
        tlen = llen + rlen
        return (llen / tlen) * H_C(lset) + (rlen / tlen) * H_C(rset)

    # Split the training set according to the index and value.
    lset, rset = split(tset, index, value)
    if len(lset) == 0 or len(rset) == 0:
        return 0

    # Calculate the information gain for this split.
    return 2.0 * (H_C(tset) - H_CS(lset, rset)) / (H_S(lset, rset) + H_C(tset))

class greedy_decision_tree(decision_tree):
    '''A very simple decision tree building algorithm based on
    the greedily maximizing the 'information gain' of a particular
    split.'''
    
    def feature_indices(self, m):
        '''Return an iterable list of feature indices. Normally
        this is just the range over 'm', but that may not always
        be the case.'''
        return range(m)
    
    def train(self, train_data):
        '''Construct a decision tree using a simple, greedy algorithm.'''
        
        m = len(train_data[0].data) # number of features

        def build_tree(node_data):
            '''Recursively construct the decision tree. '''
            node = dt_node()

            assert len(node_data) != 0

            label = node_data[0].label
            if all(item.label == label for item in node_data):
                node.label = label # leaf node (base case).
            else:
                # greedily split on information gain
                max_score = -float('inf')
                max_index = 0
                max_value = 0

                # check every feature
                for index in self.feature_indices(m):
                    # check every value of every feature
                    for item in node_data:
                        value = item.data[index]
                        score = score_test(node_data, index, value)
                        if score > max_score:
                            max_score = score
                            max_index = index
                            max_value = value
                          
                lset, rset = split(node_data, max_index, max_value)
                node.index = max_index
                node.value = max_value
                node.left = build_tree(lset)
                node.right = build_tree(rset)
            return node
    
        self.root = build_tree(train_data)

if __name__ == "__main__":
    # Basic testing code.
    from datasets import read_parkinsons_dataset
    from classifier import evaluate, normalize_dataset
    from random import shuffle
    dataset = read_parkinsons_dataset()
    shuffle(dataset)
    print('Greedy: {:.2%}'.format(evaluate(dataset, greedy_decision_tree, 4)))
    dataset = normalize_dataset(dataset)
    x = greedy_decision_tree()
    x.train(dataset)
    from simple_nn import BPNN
    from perceptron import perceptron
    print('perceptron: {:.2%}'.format(evaluate(dataset, perceptron, 4, n_features=22, n_classes=2)))
    print('BPNN: {:.2%}'.format(evaluate(dataset, BPNN, 4, n_input=22, n_hidden=22, n_output=2)))

    

