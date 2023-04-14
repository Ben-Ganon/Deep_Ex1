import numpy as np
from utils import *
from loglinear import softmax
from grad_check import gradient_check

STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}

def classifier_output(x, params):
    """
    x: a vector of features
    params: a list of the form [W, b, U, b_tag]
    """
    W, b, U, b_tag = params
    z = np.dot(x, W) + b
    h = np.tanh(z)
    o = np.dot(h, U) + b_tag
    probs = softmax(o)
    return probs

def predict(x, params):
    """
    params: a list of the form [W, b, U, b_tag]
    """
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    """
    calculates the loss and the gradients at point x with given parameters.
    params: a list of the form [W, b, U, b_tag]

    returns:
        loss,[gW, gb, gU, gb_tag]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    gU: matrix, gradients of U
    gb_tag: vector, gradients of b_tag
    """
    W, b, U, b_tag = params
    probs = classifier_output(x, params)
    loss = -np.log(probs[y])
    probs[y] -= 1
    gU = np.outer(np.tanh(np.dot(x, W) + b), probs)
    gb_tag = probs
    gb = np.dot(probs, U.T) * (1 - np.tanh(np.dot(x, W) + b) ** 2)
    gW = np.outer(x, gb)
    return loss, [gW, gb, gU, gb_tag]

def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.
    """
    W1 = np.random.randn(in_dim, hid_dim) / np.sqrt(in_dim)
    b1 = np.zeros(hid_dim)
    U = np.random.randn(hid_dim, out_dim) / np.sqrt(hid_dim)
    b2 = np.zeros(out_dim)
    return [W1, b1, U, b2]

