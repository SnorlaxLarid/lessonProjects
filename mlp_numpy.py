import os
import numpy as np
import datasets
import time


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    c = np.zeros_like(x)
    slope = 0.01
    c[x > 0] = x[x > 0]
    c[x <= 0] = slope * x[x <= 0]
    return c


def relu_deriv(x):
    x[x <= 0] = 0.01
    x[x > 0] = 1.0
    return x


def train_mlp_numpy():
    input_dim = 1024
    hidden_dim = 100
    output_dim = 10
    epochs = 1000
    lr = 0.001

    nn = NeuralNetwork(input_dim, hidden_dim, output_dim, lr)
    inputs, targets = datasets.load_datasets("digits/trainingDigits")
    number_examples = len(inputs)

    print("Training...")
    best_result = [0, 0]
    no_update = 0
    for e in range(epochs):
        init_time = time.time()
        err = 0

        for i in range(number_examples):
            error = nn.train(inputs[i], targets[i])
            err = err + error
        err = err / number_examples
        finish_time = time.time()
        diff = round((finish_time - init_time), 2)
        time_to_finish = round(((epochs - e) * diff) / 60, 2)
        print("Error: " + str(err) + " | EPOCH: " + str(e) + " | Time to finish: " + str(time_to_finish) + " mins")

        if e % 50 == 0:
            accuracy, wrong_numbers = evaluate_mlp_numpy(nn)
            if accuracy > best_result[0]:
                best_result[0] = accuracy
                best_result[1] = wrong_numbers
                no_update = 0
            else:
                no_update += 1
        if no_update >= 5:
            print("Best Accuracy on test data: " + str(best_result[0]) + "%")
            print(f"Best wrong_numbers: {best_result[1]}")
            exit()
    print("Best Accuracy on test data: " + str(best_result[0]) + "%")
    print(f"Best wrong_numbers: {best_result[1]}")


def evaluate_mlp_numpy(model):
    x, y = datasets.load_datasets("digits/testDigits")
    nn = model

    test_examples = len(x)

    ok_predictions = 0

    for i in range(test_examples):
        expected = np.argmax(y[i])
        prediction = np.argmax(nn.test(x[i]))
        if expected == prediction:
            ok_predictions += 1

    accuracy = round((ok_predictions / test_examples) * 100, 2)
    wrong_numbers = test_examples - ok_predictions
    print("Accuracy on test data: " + str(accuracy) + "%")
    print(f"wrong_numbers: {wrong_numbers}")
    return accuracy, wrong_numbers


class NeuralNetwork:

    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = learning_rate

        # np.random.seed(2231)

        self.W_I2H = np.random.rand(self.input_dim, self.hidden_dim)
        self.W_H2O = np.random.rand(self.hidden_dim, self.output_dim)

        self.B_H = np.random.rand(self.hidden_dim, 1)
        self.B_O = np.random.rand(self.output_dim, 1)

    def test(self, xlist):
        X = np.array(xlist, ndmin=2).T
        Z1 = np.dot(self.W_I2H.T, X) + self.B_H
        A1 = relu(Z1)

        Z2 = np.dot(self.W_H2O.T, A1) + self.B_O
        Y = sigmoid(Z2)

        return Y

    def train(self, xlist, tlist):
        X = np.array(xlist, ndmin=2).T
        T = np.array(tlist, ndmin=2).T

        # print("X:\n" + str(X))
        # print("T:\n" + str(T))

        ############################## FEEDFORWARD ################################

        Z1 = np.dot(self.W_I2H.T, X) + self.B_H
        A1 = relu(Z1)

        Z2 = np.dot(self.W_H2O.T, A1) + self.B_O
        Y = sigmoid(Z2)

        ############################### ERRORS ####################################

        E_Y = T - Y
        E_W1 = np.dot(self.W_H2O, E_Y)

        ############################### DELTAS ###################################

        delta_W_H2O = np.dot(-self.lr * A1, (E_Y * sigmoid_deriv(Y)).T)
        delta_B_O = -self.lr * E_Y * sigmoid_deriv(Y)

        delta_W_I2H = np.dot(-self.lr * X, (E_W1 * relu_deriv(A1)).T)
        delta_B_H = -self.lr * E_W1 * relu_deriv(A1)

        ############################## UPDATES ###################################

        self.W_H2O = self.W_H2O - delta_W_H2O
        self.B_O = self.B_O - delta_B_O

        self.W_I2H = self.W_I2H - delta_W_I2H
        self.B_H = self.B_H - delta_B_H

        return np.sum(np.absolute(E_Y))


if __name__ == "__main__":
    train_mlp_numpy()
