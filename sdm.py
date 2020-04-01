import tensorflow as tf
import numpy as np
import itertools as it


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
    def __init__(self, n, num_locations, max_distance):
        """
        :param n: Dimension of the memory
        :param num_locations: Number of locations to store registers
        :param max_distance: Given an n-dimensional cue, defines the radius within which a set of locations activate
        """
        self.n = n
        self.num_locations = num_locations
        self.max_distance = max_distance

        # Uniformly distribute registers across the n-dimensional memory space
        self.locations = np.random.randint(2, size=(num_locations, n))
        self.registers = np.zeros((num_locations, n), dtype=np.int8)

    def read(self, cue):
        # Activate the locations that are within self.max_distance of the cue
        on = np.count_nonzero(cue != self.locations, axis=1) <= self.max_distance

        # Define the output register
        out = np.zeros(self.n, dtype=np.int32)

        # Sum the registers corresponding to active locations into the output register
        np.sum(self.registers[on], out=out, axis=0)

        return out

    def write(self, cue):
        # Activate the locations that are within self.max_distance of the cue
        on = np.count_nonzero(cue != self.locations, axis=1) <= self.max_distance

        # Set the cue's off bits to -1
        cue[cue == 0] = -1

        # Edit the registers to store the new memory
        self.registers[on] += cue


mnist = tf.keras.datasets.mnist

train, test = mnist.load_data()
train = (np.array([binarize(x) for x in train[0]]), train[1])

sdm = SDM(784, 50000, 350)
point = 1000
train_x, train_y = train

# Train
for i in range(point):
    x = train_x[i]
    print(i)
    print_num(x)
    sdm.write(x.reshape(784).astype(np.int8))

# Test
for i in range(point, point + 100):
    x = train_x[i]
    print('IN')
    print_num(x)
    n = sdm.read(x.reshape(784).astype(np.int8))

    print('OUT')
    print_num(n.reshape(28, 28))
