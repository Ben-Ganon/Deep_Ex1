import numpy as np

import loglinear as ll
import random
import utils
from utils import fc as vocabulary

STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}


def feats_to_vec(features):
    # Create a numpy array of zeros with length equal to the size of the vocabulary
    vec = np.zeros(len(vocabulary))

    # Count the frequency of each word in the features and update the corresponding entry in the vector
    for word in features:
        if word in vocabulary:
            vec[vocabulary[word]] += 1

    return vec

def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        # YOUR CODE HERE
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)
        x = feats_to_vec(features)
        y = label
        y_hat = ll.predict(x, params)
        if y == y_hat:
            good += 1
        else:
            bad += 1
        pass
    return good / (good + bad)

def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for I in range(num_iterations):
        cum_loss = 0.0 # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features) # convert features to a vector.
            y = label                  # convert the label to number if needed.
            loss, grads = ll.loss_and_gradients(x,y,params)
            cum_loss += loss
            # YOUR CODE HERE
            # update the parameters according to the gradients
            # and the learning rate.
            gW = grads[0]
            gb = grads[1]
            params[0] -= learning_rate * gW
            params[1] -= learning_rate * gb



        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print(I, train_loss, train_accuracy, dev_accuracy)
    return params

if __name__ == '__main__':
    #. YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.
    train_data = utils.read_data('train')
    dev_data = utils.read_data('dev')
    in_dim = len(ll.feats_to_vec(train_data[0][1]))
    out_dim = 5
    num_iterations = 10
    learning_rate = 0.01
   
    params = ll.create_classifier(in_dim, out_dim)
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)

