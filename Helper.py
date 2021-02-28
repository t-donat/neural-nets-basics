import numpy as np
import pandas as pd
from sklearn import datasets


def sigmoid(x):
    return 1/(1+np.exp(-x))


def choose_target(labels, species="setosa"):
    if species == "setosa":
        labels[:50] = 1
        labels[50:] = 0

    elif species == "versicolor":
        labels[:50] = 0
        labels[50:100] = 1
        labels[100:] = 0

    elif species == "virginica":
        labels[:100] = 0
        labels[100:] = 1

    else:
        raise ValueError("Please enter a vaild species name: setosa, versicolor or virginica")

    return labels


def etl_iris(target="setosa", randomize=True):

    # load dataset
    iris_ds = datasets.load_iris()

    # transfer into DataFrame
    iris_data = pd.DataFrame(iris_ds.data)
    iris_data["Target"] = iris_ds.target
    iris_data.columns = ["sepal_len", "sepal_wid", "petal_len", "petal_wid", "class"]

    # split into samples X and labels Y
    iris_X = iris_data.iloc[:, :4].to_numpy()

    iris_Y = iris_data.iloc[:, 4].to_numpy()
    iris_Y = choose_target(iris_Y, species=target)
    
    # get dimensions of training set

    m, input_dim = iris_X.shape

    # randomize the order of the samples
    if randomize:
        indices = np.random.permutation(m)
        iris_X = iris_X[indices]

        iris_Y = iris_Y[indices]

    X = iris_X.T
    Y = iris_Y

    return (X, Y), (input_dim, m)

def evaluate_model(labels, predictions):
    labels = np.asarray(labels).reshape(1, -1)
    predictions = np.asarray(predictions)

    # evaluate probability predicted by the model into a label
    predicted_labels = np.where(predictions >= 0.5, 1, 0)
    correctly_predicted_labels = np.where(labels == predicted_labels, 1, 0)

    # calculate accuracy
    accuracy = np.mean(correctly_predicted_labels)

    # calculate precision
    precision = np.mean(labels[predicted_labels == 1])

    # calculate recall
    recall = np.mean(predicted_labels[labels == 1])

    # calculate F1
    F1 = 2 * precision * recall / (precision + recall)

    return accuracy, precision, recall, F1
