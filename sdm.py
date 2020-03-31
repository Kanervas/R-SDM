import random
import json
import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
import itertools as it
import os

from keras.layers import Flatten
from tensorflow_core.python.keras.layers import Dense

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"


def format_num(data):
    s = ''

    for row in data:
        for col in row:
            s += '*' if col > 0 else '-'
        s += '\n'

    return s


def print_num(data):
    print(format_num(data))


def saccadify(image: np.ndarray, n: int, saccade_length: int = 1):
    corners = it.product(range(0, len(image) - n, saccade_length), range(0, len(image[0]) - n, saccade_length))
    saccades = []

    for c in corners:
        sub_image: np.ndarray = image[c[0]:c[0] + n, c[1]:c[1] + n]
        saccades.append(sub_image.astype(np.float32))

    return np.array(saccades)


def channelify(saccades: np.array):
    shape = [x for x in saccades.shape]
    shape.append(1)
    return saccades.reshape(shape)


def binarize(l: np.ndarray):
    z = l.flatten()
    z[z > 0] = 1
    return z.astype(np.float32).reshape((28, 28))


class SDM:
    def __init__(self, n, l, d):
        self.n = n
        self.l = l
        self.d = d

        # Place l n-dimensional locations uniformally throughout the space
        self.places = np.random.randint(2, size=(l, n))
        self.registers = np.zeros((l, n), dtype=np.int8)

    def hamming_distance(self, v1, v2):
        return np.count_nonzero(v1 != v2)

    def read(self, m):
        on = np.count_nonzero(m != self.places, axis=1) <= self.d
        # print('read-on', m, on)
        # print('regs-on', self.registers[on])
        # print('read-sum', m, np.sum(self.registers[on], axis=0))
        out = np.zeros(self.n, dtype=np.int32)
        np.sum(self.registers[on], out=out, axis=0)
        # print('read-out', out)

        return out

    def write(self, m):
        on = np.count_nonzero(m != self.places, axis=1) <= self.d
        # print('write-on', m, on)
        m[m == 0] = -1
        self.registers[on] += m
        # print('write-regs', m, self.registers)


def to_num(arr):
    s = 0
    for i in range(len(arr)):
        if arr[i] > 0:
            s += 2 ** i

    return s


def to_arr(n):
    arr = np.zeros(784, dtype=np.int8)
    for i in range(784):
        if n % 2 ** i == 0:
            arr[i] = 1

    return arr


mnist = tf.keras.datasets.mnist

train, test = mnist.load_data()
print(train)

train = (np.array([binarize(x) for x in train[0]]), train[1])
print(len(train), len(test))
print(train[0][0])

sdm = SDM(784, 50000, 350)
c = 0
point = 1000
for x in train[0]:
    print(c)
    print_num(x)

    sdm.write(x.reshape(784).astype(np.int8))
    if c > point:
        break
    else:
        c += 1

for t in train[0][point:point + 100]:
    print('IN')
    print_num(t)
    n = sdm.read(t.reshape(784).astype(np.int8))
    # print('1')
    print_num(n.reshape(28, 28))
    n = sdm.read(n.reshape(784).astype(np.int8))
    # print('2')
    print_num(n.reshape(28, 28))
    n = sdm.read(n.reshape(784).astype(np.int8))
    # print('3')
    print_num(n.reshape(28, 28))
    n = sdm.read(n.reshape(784).astype(np.int8))
    # print('4')
    print_num(n.reshape(28, 28))
    n = sdm.read(n.reshape(784).astype(np.int8))
    # print('5')
    print_num(n.reshape(28, 28))
    n = sdm.read(n.reshape(784).astype(np.int8))
    # print('6')
    print_num(n.reshape(28, 28))

    print('OUT')
    print_num(n.reshape(28, 28))
