import numpy as np
from loglinear import softmax

STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}

def classifier_output(x, params):
    temp_x = x
    for index in range(0,len(params),2):
        temp_x = np.tanh(np.dot(temp_x, params[index]) + params[index+1])
    probs = softmax(temp_x)
    return probs

def classifier_output_with_cache(x, params):
    temp_x = x
    cache = []
    for index in range(0,len(params),2):
        cache.append(temp_x)
        temp_x = np.tanh(np.dot(temp_x, params[index]) + params[index+1])
    probs = softmax(temp_x)
    return probs, cache

def predict(x, params):
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    """
    params: a list as created by create_classifier(...)

    returns:
        loss,[gW1, gb1, gW2, gb2, ...]

    loss: scalar
    gW1: matrix, gradients of W1
    gb1: vector, gradients of b1
    gW2: matrix, gradients of W2
    gb2: vector, gradients of b2
    ...

    (of course, if we request a linear classifier (ie, params is of length 2),
    you should not have gW2 and gb2.)
    """
    probs, cache = classifier_output_with_cache(x, params)
    loss = -np.log(probs[y])
    probs[y] -= 1
    grads = []
    # delet all the bias from params
    params = [p for index,p in enumerate(params) if index%2==0 ]
    for index in range(len(params)-1, -1, -1):
        grads.append(probs)
        grads.append(np.outer(cache[index], probs))
        probs = np.dot(probs, params[index].T) * (1 - cache[index] ** 2)
    grads.reverse()
    return loss, grads

def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.
    
    Assume a tanh activation function between all the layers.

    return:
    a flat list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """

    params = []
    for i in range(len(dims) - 1):
        if dims[i] < 1:
            raise ValueError('dims must be a list of positive integers')
        W = np.random.randn(dims[i], dims[i+1]) / np.sqrt(dims[i])
        b = np.zeros(dims[i+1])
        params.append(W)
        params.append(b)
    return params

