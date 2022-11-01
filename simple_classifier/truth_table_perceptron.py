import numpy as np
from tqdm import trange


def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_derivative(x):
    return x*(1-x)


def check_accuracy(test, pred):

    return np.mean(test == pred)


class Perceptron:

    def __init__(self, num_inputs, epochs=100, learning_rate=0.01):

        self.epochs = epochs
        self.learning_rate = learning_rate

        # index 0 contains the bias, the rest are weights
        self.parameters = 2*np.random.random((num_inputs+1, 1))
        self.train_features = None

    def load_data(self, data):

        data = np.asarray(data)
        print(data)

        self.train_features = np.hstack(np.ones((len(data), 1), data))

    def predict(self, inputs):
        summation = np.dot(inputs, self.parameters)

        summation[summation > 0] = 1
        summation[summation < 0] = 0

        return summation

    def fit(self, training_inputs, training_labels):
        print(f"Starting training for {self.epochs} epochs.")

        for _ in trange(self.epochs):

            for feature, label in zip(training_inputs, training_labels):
                prediction = self.predict(feature)

                result = self.learning_rate * (label - prediction) * feature

                self.parameters += result.reshape((-1, 1))

        print("Training completed.")


if __name__ == "__main__":

    train_features = np.array([[0, 0, 0],
                               [0, 0, 1],
                               [0, 1, 0],
                               [0, 1, 1],
                               [1, 0, 0],
                               [1, 0, 1],
                               [1, 1, 0],
                               [1, 1, 1]])

    train_features_bias = np.hstack((np.ones((len(train_features), 1)), train_features))

    train_labels = np.array([0, 1, 1, 1, 0, 1, 1, 1])

    perceptron = Perceptron(3, epochs=10000)
    perceptron.fit(train_features_bias, train_labels)

    results = perceptron.predict(train_features_bias)

    print(results)

    accuracy = check_accuracy(train_labels, results[:, 0])

    print(accuracy)
