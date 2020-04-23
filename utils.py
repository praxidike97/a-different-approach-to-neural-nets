import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def generate_linear_separable_1d_distribution(length=1000, plot=False):
    values01 = np.random.normal(loc=4, scale=1, size=(length, 1))
    values02 = np.random.normal(loc=10, scale=1, size=(length, 1))

    if plot:
        plt.scatter(values01, np.zeros((length, )))
        plt.scatter(values02, np.ones((length,)))
        plt.show()

    xs = np.concatenate((values01, values02))
    ys = np.concatenate((np.zeros((length, )).astype(int), np.ones((length, )).astype(int)))
    indices = np.arange(start=0, stop=length*2)
    np.random.shuffle(indices)

    xs = xs[indices]
    ys = ys[indices]

    return xs, ys


def generate_linear_separable_2d_distribution(length=1000, plot=False):
    xs, ys = list(), list()

    values01 = np.random.uniform(low=0, high=1, size=(length, 1))
    values02 = np.random.uniform(low=0, high=1, size=(length, 1))

    for value01, value02 in zip(values01, values02):
        if value02 > value01:
            ys.append(1)
        else:
            ys.append(0)

        xs.append([value01[0], value02[0]])

    xs = np.asarray(xs)
    ys = np.asarray(ys)

    if plot:
        color = np.array(["r", "b"])
        plt.scatter(xs[:, 0], xs[:, 1], color=color[ys])
        plt.show()

    return xs, ys


def generate_linear_unseparable_2d_distribution(length=1000, plot=False, include_squares=False):
    xs, ys = list(), list()

    values01 = np.random.uniform(low=-2.5, high=2.5, size=(length, 1))
    values02 = np.random.uniform(low=0, high=3, size=(length, 1))

    for value01, value02 in zip(values01, values02):
        if -np.sqrt(value02) < value01 < np.sqrt(value02):
            ys.append(1)
        else:
            ys.append(0)

        if include_squares:
            xs.append([value01[0], value02[0], value01[0]**2])
        else:
            xs.append([value01[0], value02[0]])

    xs = np.asarray(xs)
    ys = np.asarray(ys)

    if plot:
        color = np.array(["r", "b"])
        plt.scatter(xs[:, 0], xs[:, 1], color=color[ys])
        plt.show()

    return xs, ys


def to_one_hot(labels, num_classes=2):
    targets = np.array(labels).reshape(-1)
    return np.eye(num_classes)[targets]


def normalize(x):
    return (x - np.mean(x))/np.std(x)


def newline(p1, p2):
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()

    if(p2[0] == p1[0]):
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])

    l = mlines.Line2D([xmin,xmax], [ymin,ymax])
    ax.add_line(l)
    return l
