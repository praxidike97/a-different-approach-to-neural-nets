import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import generate_linear_separable_2d_distribution, generate_linear_separable_1d_distribution, generate_linear_unseparable_2d_distribution, \
    normalize, to_one_hot


def softmax_function(x, weights):
    probabilities = list()

    overall_sum = sum([np.exp(np.dot(x, weights[i])) for i in range(weights.shape[0])])
    for i in range(weights.shape[0]):
        alpha = np.dot(weights[i], x)
        probabilities.append(np.exp(alpha)/overall_sum)

    return np.asarray(probabilities)


def train_multinomial(xs, ys, weights, lr=0.3, epochs=500, num_classes=2):
    for e in tqdm(range(epochs)):
        sum_over_samples = np.ones(shape=(num_classes, weights.shape[1]))
        for x, y in zip(xs, ys):
            x = np.append(x, 1.)
            P = np.repeat(softmax_function(x, weights)[:, np.newaxis], repeats=weights.shape[1], axis=1)
            ones = np.ones(shape=(num_classes, weights.shape[1]))

            x = np.repeat(x[np.newaxis, :], repeats=num_classes, axis=0)
            y = np.repeat(y[:, np.newaxis], repeats=weights.shape[1], axis=1)

            sum_over_samples += y*(ones-P)*x
            #sum_over_samples += (y - P) * x

        weights += lr*(1/len(xs))*sum_over_samples

    return weights


def predict_multinomial(xs, weights):
    predictions = list()

    for x in xs:
        x = np.append(x, 1.)
        prediction = np.argmax(softmax_function(x, weights))
        predictions.append(prediction)

    return predictions


def train_binary(xs, ys, lr=0.3, epochs=500, num_features=3):
    weights = np.zeros(num_features + 1)
    for e in tqdm(range(epochs)):
        delta_weights = np.zeros(shape=(num_features + 1, ))
        for x, y in zip(xs, ys):
            x = np.append(x, 1.)
            sigmoid_value = 1/(1 + np.exp(-np.dot(x, weights)))

            delta_weights += x*(y - sigmoid_value)

        weights += lr*(1/len(xs))*delta_weights

    return weights


def predict_binary(xs, weights):
    predictions = list()

    for x in xs:
        x = np.append(x, 1.)
        prediction = 1/(1 + np.exp(-np.dot(x, weights)))

        if prediction < 0.5:
            predictions.append(0)
        else:
            predictions.append(1)

    return predictions


def create_decision_boundary(xs, ys, weights):
    xx, yy = np.meshgrid(np.arange(-2.5, 2.5, 0.01), np.arange(0.0, 3.0, 0.01))
    xx02 = np.power(xx, 2)
    values = np.c_[xx.ravel(), yy.ravel(), xx02.ravel()]
    Z = np.asarray(predict_binary(values, weights))
    Z = Z.reshape(xx.shape)

    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot also the training points
    scatter = plt.scatter(xs[:, 0], xs[:, 1], c=ys, edgecolors='k', cmap=plt.cm.Paired)
    plt.legend(handles=scatter.legend_elements()[0], labels=["Class 1", "Class 2"], loc="upper left")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    #plt.xticks(())
    #plt.yticks(())
    plt.show()


def plot_sigmoid_function(xs, ys, weights):
    values = np.arange(-10.0, 20., 0.01)
    values = np.array([np.array([value, 1.]) for value in values])

    predictions = np.array([softmax_function(value, weights) for value in values])
    plt.scatter(values[:, 0], predictions[:, 0])

    plt.scatter(xs, ys, c=ys, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('petal_width')
    plt.ylabel('sepal_length')

    plt.show()


def plot_data(xs, ys):
    scatter = plt.scatter(xs[:, 0], xs[:, 1], c=ys, edgecolors='k', cmap=plt.cm.Paired)
    plt.legend(handles=scatter.legend_elements()[0], labels=["Class 1", "Class 2"], loc="upper left")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


if __name__ == "__main__":
    data = pd.read_csv("./data/iris.csv")
    feature_names = data.columns.values

    #xs_raw = data[feature_names[:-1]].values
    #xs = normalize(xs_raw)

    #ys_raw = data[feature_names[-1]].values

    num_classes = 2#len(np.unique(ys_raw))
    num_features = 3#xs.shape[1]

    #ys = to_one_hot(ys_raw, num_classes=num_classes)


    xs, ys = generate_linear_unseparable_2d_distribution(include_squares=True)
    #xs, ys = generate_linear_separable_2d_distribution()
    #ys = to_one_hot(ys_raw, num_classes=num_classes)
    plot_data(xs, ys)

    weights = np.zeros((num_classes, num_features + 1))
    #results = softmax_function(np.array(xs[0]), weights)

    weights = train_binary(xs, ys, epochs=500, num_features=num_features)
    predictions = predict_binary(xs, weights)

    #weights = train_multinomial(xs, ys, weights, num_classes=num_classes)
    #predictions = predict_multinomial(xs, weights)

    correct_predictions = 0
    for prediction, y_true in zip(predictions, ys):
        if prediction == y_true:
            correct_predictions +=1

    print("Accuracy: %f" % (correct_predictions / len(xs)))

    create_decision_boundary(xs, ys, weights)
    #plot_sigmoid_function(xs, ys_raw, weights)