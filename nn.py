import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K

from utils import generate_linear_unseparable_2d_distribution, to_one_hot, newline

NUM_HIDDEN = 1


def create_model(input_size, output_size):
    model = Sequential()
    model.add(Dense(NUM_HIDDEN, activation="sigmoid", input_shape=(input_size, )))
    model.add(Dense(output_size, activation="softmax"))

    model.compile("adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def create_decision_boundary(xs, ys, model):
    xx, yy = np.meshgrid(np.arange(-2.5, 2.5, 0.01), np.arange(0.0, 3.0, 0.01))
    values = np.c_[xx.ravel(), yy.ravel()]

    Z = np.round(np.asarray(model.predict(values)[:, 0]))
    #Z = np.asarray(model(values))[0, :, 0]

    """
    fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
    plots = list()

    for i in range(0, NUM_HIDDEN):
        #Z += np.asarray(model(values))[0, :, i]
        Z = np.asarray(model(values))[0, :, i]
        Z = Z.reshape(xx.shape)

        #plt.subplot(5, 10, i+1, sharex=True, sharey=True)
        ax = axs[i%2]
        plot = ax.pcolormesh(xx, yy, Z, cmap="plasma", vmin=0.0, vmax=1.0)
        plots.append(plot)
        #ax.xlabel("Feature 1")
        #ax.ylabel("Feature 2")
        #plt.show()
        #Z += np.round(np.asarray(model.predict(values)[:, i]))

    fig.text(0.5, 0.02, 'Feature 1', ha='center')
    fig.text(0.06, 0.5, 'Feature 2', va='center', rotation='vertical')

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.04, 0.7])
    fig.colorbar(plots[0], cax=cbar_ax)
        plt.show()
    """

    Z = Z.reshape(xx.shape)

    #plt.pcolormesh(xx, yy, Z, cmap="plasma")

    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
    #plt.colorbar()

    #newline([0, -0.395], [-3, 4.474])

    # Plot also the training points
    ys = np.argmax(ys, axis=1)
    scatter = plt.scatter(xs[:, 0], xs[:, 1], c=np.ones(len(ys))-ys, edgecolors='k', cmap=plt.cm.Paired)
    plt.legend(handles=scatter.legend_elements()[0], labels=["Class 1", "Class 2"], loc="upper left")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()


if __name__ == "__main__":
    num_classes = 2
    num_features = 2

    xs, ys_raw = generate_linear_unseparable_2d_distribution()
    ys = to_one_hot(ys_raw, num_classes=num_classes)

    model = create_model(input_size=num_features, output_size=num_classes)
    model.fit(x=xs, y=ys, epochs=500)

    create_decision_boundary(xs, ys, model)

    get_1st_layer_output = K.function([model.layers[0].input],
                                      [model.layers[0].output])

    print("Number of layers: %i" % len(model.layers))
    print("Weight outputs: %s" % str(model.layers[-1].get_weights()))
    print("Weight outputs: %s" % str(model.layers[0].get_weights()))

    #create_decision_boundary(xs, ys, get_1st_layer_output)