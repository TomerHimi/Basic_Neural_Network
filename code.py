"""
Created on Sat Dec 12 14:02:14 2020
@author: Tomer Himi

"""
import numpy as np
import random
from scipy.special import expit
np.random.seed(8)

eta = 0.04
epochs = 100
features_num = 784
labels_num = 10
sizes = [784, 40 ,10]
mini_batch_size = 28
BIASES = [np.random.randn(y, 1) for y in sizes[1:]]
WEIGHTS = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

def normalization(data):
    """pre-proccesing function of raw data
    parm: data: array of training data
    type: data: ndarray
    return: normalized data for learning
    rtype: data: ndarray"""
    global features_num
    for row in data:
        for i in range(features_num):
            row[i] = row[i] / 255.
    return data

def vectorized_label(label):
    """pre-proccesing function of raw data
    parm: label: array of lables for training data
    type: label: ndarray
    return: formmated array of labels for learning
    rtype: data: ndarray"""
    vec = np.zeros((labels_num, 1))
    vec[int(label)] = 1.0
    return vec

def train_logistic_regression(training_data,test_data):
    """implementation function of logistic regression
    parm: training_data: array of training data
    parm: test_data: array of testing data
    type: training_data: ndarray
    type: test_data: ndarray"""
    global mini_batch_size
    n = len(training_data)
    for i in range(epochs):
        random.shuffle(training_data)
        mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
        for mini_batch in  mini_batches:
            delta_b = update_mini_batches(mini_batch)
            
def update_mini_batches(mini_batch):
    """implementation function of neural network
    parm: mini_batch: array of training data
    type: mini_batch: ndarray
    return: BIASES array
    rtype: ndarray"""
    global WEIGHTS
    global BIASES
    nabla_b = [np.zeros(b.shape) for b in BIASES]
    nabla_w = [np.zeros(w.shape) for w in WEIGHTS]
    for x, y in mini_batch:
        delta_nabla_b, delta_nabla_w = backprop(x, y)
        nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
    WEIGHTS = [w - (eta / len(mini_batch)) * nw for w, nw in zip(WEIGHTS, nabla_w)]
    BIASES = [b - (eta / len(mini_batch)) * nb for b, nb in zip(BIASES, nabla_b)]
    return np.multiply(nabla_b, 1.0 / len(mini_batch))

def backprop(x, y):
    """implementation function of back propagation
    parm: x: one example from training data
    parm: y: label of x from training data
    type: x: ndarray 
    type: y: ndarray
    return: derivatives of the parameters of the model
    rtype: tuple"""
    w = {}
    b = {}
    z = {}
    a = {}
    for k in range(1, len(WEIGHTS) + 1):
        w[k] = WEIGHTS[k - 1]
        b[k] = BIASES[k - 1]
    a[0] =  np.expand_dims(x, axis = 1)
    z[0] = np.expand_dims(x, axis = 1)
    #forward propagation
    for k in range(1, len(w) + 1):
        z[k] = np.dot(w[k], a[k - 1]) + b[k]
        if k == len(w):
            a[k] = softmax(z[k])
        else:
            a[k] = sigmoid(z[k])
    #back propagation
    delta = {}
    for k in range(len(w), 0, -1):
        if k == len(w):
            delta[len(w)] = a[len(w)] - np.transpose(y)
        else:
            first_expression = np.dot(np.transpose(w[k + 1]), delta[k + 1])
            second_expression = derivative_sigmoid(z[k])
            delta[k] = np.multiply(first_expression, second_expression)
    db = []
    dw = []
    for k in range(1, len(w) + 1):
        db.append(delta[k])
        dw.append(np.dot(delta[k], np.transpose(a[k - 1])))
    return db, dw

def softmax(z):
    """calculation function of softmax
    parm: z: vector of numbers
    type: z: ndarray
    return: vector of probabilities 
    rtype: ndarray"""
    z -= np.max(z)
    output_exp = np.exp(z)
    return output_exp / output_exp.sum()

def sigmoid(z):
    """calculation of sigmoid function 
    parm: z: vector of numbers
    type: z: ndarray
    return: result of sigmoid function on a given vector
    rtype: ndarray"""
    return expit(z)

def derivative_sigmoid(z):
    """calculation of derivative_sigmoid function
    parm: z: vector of numbers
    type: z: ndarray
    return: result of derivative sigmoid on a given vector
    rtype: ndarray"""
    return expit(z) * (1 - expit(z))

def test_labels(test_data):
    """testing step for logistic regression
    parm: test_data: array of testing data
    type: test_data: ndarray
    return: prediction vector of lables for test data
    rtype: ndarray"""
    output_result = [np.argmax(output_before_softmax(x)) for x in test_data]
    return output_result

def output_before_softmax(data_x):
    """testing step before using softmax function
    parm: data_x: one example from test data
    type: data_x: ndarray
    return: output of neural network before using softmax
    rtype: ndarray"""
    x = np.expand_dims(data_x, axis = 1)
    layer = 0
    for b, w in zip(BIASES, WEIGHTS):
        if layer == len(WEIGHTS) - 1:
            x = np.dot(w, x) + b
        else:
            x = sigmoid(np.dot(w,x) + b)
        layer += 1
    return x

def main():
    train_x = np.loadtxt("train_x")
    train_y = np.loadtxt("train_y")
    test_data = np.loadtxt("test_x")
    #pre-proccesing 
    normalized_data = normalization(train_x)
    labels = [vectorized_label(y) for y in train_y]
    labels = [np.reshape(x, (1, labels_num)) for x in labels]
    data = list(zip(normalized_data, labels))
    training_data = data[0:50000]
    validation_data = data[50000:55000]  #5000 examples for validation
    #training step
    train_logistic_regression(training_data, validation_data)
    #testing step
    result = test_labels(test_data)
    a = np.asarray(result)
    np.savetxt("test_y", a, fmt = "%d", delimiter = "")

if __name__ == '__main__':
    main()