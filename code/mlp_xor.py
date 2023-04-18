import numpy as np

import mlp1 as ll
import random
import utils
from utils import F_UNI_I as vocabulary
from xor_data import data
STUDENT = {'name1': 'Omri Ben Hemo',
           'ID1': '313255242',
           'name2': 'Ben Ganon',
           'ID2': '318731007'
           }


def feats_to_vec(features):
   return features

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
    max_dev_acc = 0
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
    return params, dev_accuracy


def get_key_from_dict(dict, value):
    for key, val in dict.items():
        if val == value:
            return key


def run_on_test_and_write(params):
    """
    Read the test set, predict the labels, and write the output to a file.
    """
    test_data = utils.read_data('../data/test')
    with open('test.pred', 'w') as f:
        for label, features in test_data:
            x = feats_to_vec(features)
            y_hat = ll.predict(x, params)
            print(get_key_from_dict(utils.L2I, y_hat), file=f)

if __name__ == '__main__':
    #. YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.
    train_data = data
    dev_data = data
    in_dim = len(feats_to_vec(train_data[0][1]))
    hid_dim = 16
    out_dim = 6
    num_iterations = 50
    learning_rate = 0.1
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
