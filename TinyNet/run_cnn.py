# vim: expandtab:ts=4:sw=4
import numpy as np
from utils.utils import load_mnist
from modules.layers import *
from modules.solver import sgd, sgd_momentum, adam
from modules.nnet import CNN
import sys


def make_mnist_cnn(X_dim, num_class):
    conv = Conv(X_dim, n_filter=32, h_filter=3,
                w_filter=3, stride=1, padding=1)
    relu_conv = ReLU()
    maxpool = Maxpool(conv.out_dim, size=2, stride=1)
    flat = Flatten()
    fc = FullyConnected(np.prod(maxpool.out_dim), num_class)
    return [conv, relu_conv, maxpool, flat, fc]


def make_cifar10_cnn(X_dim, num_class):
    conv = Conv(X_dim, n_filter=16, h_filter=5,
                w_filter=5, stride=1, padding=2)
    relu = ReLU()
    maxpool = Maxpool(conv.out_dim, size=2, stride=2)
    conv2 = Conv(maxpool.out_dim, n_filter=20, h_filter=5,
                 w_filter=5, stride=1, padding=2)
    relu2 = ReLU()
    maxpool2 = Maxpool(conv2.out_dim, size=2, stride=2)
    flat = Flatten()
    fc = FullyConnected(np.prod(maxpool2.out_dim), num_class)
    return [conv, relu, maxpool, conv2, relu2, maxpool2, flat, fc]


if __name__ == "__main__":

    trainset = load_mnist('data/train-images-idx3-ubyte.gz', 'data/train-labels-idx1-ubyte.gz')
    testset = load_mnist('data/t10k-images-idx3-ubyte.gz', 'data/t10k-labels-idx1-ubyte.gz')
	shape = (-1,1,28,28)
	X, y = trainset
	X = X.reshape(shape)
	X_test, y_test = testset
	X_test = X_test.reshape(shape)
    mnist_dims = (1, 28, 28)
    cnn = CNN(make_mnist_cnn(mnist_dims, num_class=10))
    cnn = sgd_momentum(cnn, X, y, minibatch_size=32, epoch=20,
                           learning_rate=0.01, X_test=X_test, y_test=y_test)

