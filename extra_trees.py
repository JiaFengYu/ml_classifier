from classifier import classifier
from decision_tree import decision_tree, dt_node, score_test, split
from bagging import bagging_trees
from random import sample, uniform

def majority_label(dataset):
    '''Return the most common label in a dataset.'''
    counts = {}
    for item in dataset:
        counts[item.label] = counts.get(item.label, 0) + 1
    return max(counts.keys(), key=lambda x: counts[x])

class extra_tree(decision_tree):
    '''Build a tree using the 'extremely randomized trees' algorithm.'''
    def train(self, training_data, K = 10, Nmin = 2):
        '''Train an individual extra tree.'''
        
        def same_labels(node_data):
            '''Return True if all labels in the training data
            are the same.'''
            label = node_data[0].label
            return all(label == x.label for x in node_data[1:])
        
        def non_constant_features(node_data):
            '''Return a list of features that are not constant in the
            training data.'''
            indices = []
            m = len(node_data[0].data)
            # Check each feature
            for i in range(m):
                value = node_data[0].data[i]
                # Compare all feature values to the first value.
                if any(value != x.data[i] for x in node_data[1:]):
                    # Not constant, add it to the list.
                    indices.append(i)
            return indices

        def pick_random_split(node_data, indices):
            '''Pick the best split of the K random splits generated.'''
            max_score = -float('inf')
            max_index = 0
            max_value = 0
            m = len(node_data[0].data)
            for index in indices:
                feature = [item.data[index] for item in node_data]
                value = uniform(min(feature), max(feature))
                score = score_test(node_data, index, value)
                if score > max_score:
                    max_score = score
                    max_index = index
                    max_value = value
            return (max_index, max_value)

        def build_tree(node_data, K, Nmin):
            '''Recursively build a tree using the extra tree algorithm.'''
            node = dt_node()
            n = len(node_data)
            indices = non_constant_features(node_data)
            if n < Nmin or len(indices) == 0 or same_labels(node_data):
                node.label = majority_label(node_data)
            else:
                if len(indices) > K:
                    indices = sample(indices, K)
                node.index, node.value = pick_random_split(node_data, indices)
                left_data, right_data = split(node_data, node.index, node.value)
                node.left = build_tree(left_data, K, Nmin)
                node.right = build_tree(right_data, K, Nmin)
            return node

        self.root = build_tree(training_data, K, Nmin)
        
class extra_trees(bagging_trees):
    '''Implement "Extremely Randomized trees", the random forest classifier
    described in Geurts et al.  2006.'''
    def __init__(self, M = 15, K = 10, Nmin = 2):
        '''Initialize the empty forest for an 
        Extremely Randomized ("extra") tree classifier.'''
        super().__init__(M)
        self.K = K
        self.Nmin = Nmin

    def train(self, training_data):
        '''Train a random forest using the 'extremely randomized'
        tree algorithm.'''
        for i in range(self.M):
            dt = extra_tree()
            dt.train(training_data, self.K, self.Nmin)
            self.forest.append(dt)
            
if __name__ == "__main__":
    from datasets import read_parkinsons_dataset
    from classifier import evaluate
    from random import shuffle
    dataset = read_parkinsons_dataset()
    shuffle(dataset)
    m = len(dataset[0].data)    # number of features
    pct = evaluate(dataset, extra_trees, 4, M=99, K=int(m ** 0.5))
    print('Extra Trees: {:.2%}'.format(pct))

