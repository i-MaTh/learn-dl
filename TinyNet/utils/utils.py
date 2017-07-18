import numpy as np
import pickle
import gzip, struct
import os


def one_hot_encode(y, num_class):
    m = y.shape[0]
    onehot = np.zeros((m, num_class), dtype="int32")
    for i in range(m):
        idx = y[i]
        onehot[i][idx] = 1
    return onehot


def accuracy(y_true, y_pred):
    return np.mean(y_pred == y_true)  # both are not one hot encoded


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def load_mnist(image_file, label_file):
	"""
	read data into numpy
	"""
	
	with gzip.open(label_file) as flbl:
		magic, num = struct.unpack(">II", flbl.read(8))
		label = np.fromstring(flbl.read(), dtype=np.int8)
	with gzip.open(image_file) as fimg:
		magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
		image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
	return (image, label)
	

