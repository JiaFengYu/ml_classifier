from classifier import *
from datasets import *
from random import shuffle, seed

def dot_product(x, y):
    '''Compute the dot product (scalar product) for two vectors.'''
    return sum(a * b for a, b in zip(x, y))

class perceptron(classifier):
    '''Simple implementation of a multiclass perceptron.
    
    In a multiclass perceptron, there is one weight vector per class.
    '''
    def __init__(self, n_features, n_classes):
        '''Initialize a multiclass perceptron with 'n_features'
        dimensions and 'n_classes' output classifications.'''
        self.n_features = n_features + 1 # additional bias term.
        self.n_classes = n_classes
        self.weights = []
        for i in range(self.n_classes):
            self.weights.append([0] * self.n_features)

    def train(self, train_data, iterations = 1000):
        '''Training phase for multiclass perceptron.'''
        for iter in range(iterations):
            n_errors = 0
            for item in train_data:
                pred_label, feature_vector = self.__predict(item.data)
                if item.label != pred_label:
                    n_errors += 1
                    for i in range(self.n_features):
                        try:
                            self.weights[item.label][i] += feature_vector[i]
                            self.weights[pred_label][i] -= feature_vector[i]
                        except IndexError:
                            print(item.label, i, len(item.data))
            if n_errors == 0:
                break
    
    def __predict(self, data):
        '''Return the most likely class label for an unknown data point.'''
        inputs = data + [1]     # bias term
        activations = [dot_product(inputs, self.weights[c]) for c in range(self.n_classes)]
        predicted_label = argmax(activations)
        return predicted_label, inputs

    def predict(self, data):
        result = self.__predict(data)
        return result[0]

# Main program
if __name__ == "__main__":
    seed(0)                     # Force RNG to known state.

    iris_dataset = read_iris_dataset()
    shuffle(iris_dataset)
    wine_dataset = read_wine_dataset()
    shuffle(wine_dataset)
    wheat_dataset = read_seeds_dataset()
    shuffle(wheat_dataset)

    print("Testing the perceptron.")
    print()
    print("The iris dataset.")
    print("With raw data:")
    print("accuracy:", evaluate(iris_dataset, perceptron, 4, n_features=4, n_classes=3))
    print("With normalized data:")
    print("accuracy:", evaluate(normalize_dataset(iris_dataset), perceptron, 4, n_features=4, n_classes=3))
    print()
    print("The wheat dataset.")
    print("With raw data:")
    print("accuracy:", evaluate(wheat_dataset, perceptron, 4, n_features=7, n_classes=3))
    print("With normalized data:")
    print("accuracy:", evaluate(normalize_dataset(wheat_dataset), perceptron, 4, n_features=7, n_classes=3))
    print()
    print("The wine dataset.")
    print("With raw data:")
    print("accuracy:", evaluate(wine_dataset, perceptron, 4, n_features=13, n_classes=3))
    print("With normalized data:")
    print("accuracy:", evaluate(normalize_dataset(wine_dataset), perceptron, 4, n_features=13, n_classes=3))
    print()
    print("The xor dataset.")
    xor = [data_item(0, [0.0, 0.0]),
           data_item(1, [1.0, 0.0]),
           data_item(1, [0.0, 1.0]),
           data_item(0, [1.0, 1.0])]

    p = perceptron(2, 2)
    p.train(xor, 2000)
    n_correct = 0
    for item in xor:
        label = p.predict(item.data)
        if label == item.label:
            n_correct += 1
    print("accuracy: ", n_correct / len(xor))
