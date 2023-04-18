import numpy as np

import mlp1 as ll
import random
import utils
from utils import F2I as vocabulary

STUDENT = {'name1': 'Omri Ben Hemo',
           'ID1': '313255242',
           'name2': 'Ben Ganon',
           'ID2': '318731007'
           }

def read_data(fname):
    data = []
    with open(fname, encoding='utf-8') as f:
        for line in f:
            label, text = line.strip().lower().split("\t",1)
            data.append((label, text))
    return data

def feats_to_vec(features):
    # Create a numpy array of zeros with length equal to the size of the vocabulary
    vec = np.zeros(len(vocabulary))
    # Count the frequency of each word in the features and update the corresponding entry in the vector
    for word in utils.text_to_bigrams(features):
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
        y = utils.L2I[label]
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
    max_dev_acc = 0
    for I in range(num_iterations):
        cum_loss = 0.0 # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features) # convert features to a vector.
            y = utils.L2I[label]                  # convert the label to number if needed.
            loss, grads = ll.loss_and_gradients(x,y,params)
            cum_loss += loss
            # YOUR CODE HERE
            # update the parameters according to the gradients
            # and the learning rate.
            gW = grads[0]
            gB = grads[1]
            gU = grads[2]
            gb_tag = grads[3]
            params[0] -= learning_rate * gW
            params[1] -= learning_rate * gB
            params[2] -= learning_rate * gU
            params[3] -= learning_rate * gb_tag

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)

        print(I, train_loss, train_accuracy, dev_accuracy)
    return params


def get_key_from_dict(dict, value):
    for key, val in dict.items():
        if val == value:
            return key


def run_on_test_and_write(params):
    """
    Read the test set, predict the labels, and write the output to a file.
    """
    test_data = read_data('../data/test')
    with open('test.pred', 'w') as f:
        for label, features in test_data:
            x = feats_to_vec(features)
            y_hat = ll.predict(x, params)
            print(get_key_from_dict(utils.L2I, y_hat), file=f)

if __name__ == '__main__':
    #. YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.
    train_data = utils.read_data('../data/train')
    dev_data = utils.read_data('../data/dev')
    in_dim = len(feats_to_vec(train_data[0][1]))
    hid_dim = 18
    out_dim = 6
    num_iterations = 50
    learning_rate = 0.001
    # l_rates = [0.001, 0.01,0.05, 0.2, 0.3, 0.1, 0.5]
    # hid_dims = [8, 16, 32, 64]
    # for rate in l_rates:
    #     for dim in hid_dims:
    #         print("learning rate: ", rate, " hidden dim: ", dim)
    #         params = ll.create_classifier(in_dim, dim, out_dim)
    #         trained_params, dev_acc = train_classifier(train_data, dev_data, num_iterations, rate, params)
    #         print("dev accuaracy: ", dev_acc)

    params = ll.create_classifier(in_dim, hid_dim, out_dim)
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)
    run_on_test_and_write(trained_params)
