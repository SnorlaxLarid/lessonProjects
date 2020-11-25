import numpy as np
import datasets
from sklearn.neural_network import MLPClassifier


def tran_mlp_sklearn():
    input_dim = 32 * 32
    hidden_dim = 200
    output_dim = 10
    lr = 0.001
    train_data, train_label = datasets.load_datasets("digits/trainingDigits")
    train_label = [np.argmax(label) for label in train_label]
    train_data = train_data.reshape(train_data.shape[0], train_data.shape[2])
    model = MLPClassifier(hidden_layer_sizes=(hidden_dim,), activation="tanh", solver="sgd", learning_rate_init=lr,
                          max_iter=1000)
    model.fit(train_data, train_label)

    test_data, test_label = datasets.load_datasets("digits/testDigits")
    test_label = [np.argmax(label) for label in test_label]
    test_data = test_data.reshape(test_data.shape[0], test_data.shape[2])
    predictions = model.predict(test_data)
    ok_predictions = 0
    for i in range(len(predictions)):
        expected = test_label[i]
        prediction = predictions[i]
        if expected == prediction:
            ok_predictions += 1

    accuracy = round((ok_predictions / len(predictions)) * 100, 2)
    wrong_numbers = len(predictions) - ok_predictions
    print("Accuracy on test data: " + str(accuracy) + "%")
    print(f"wrong_numbers: {wrong_numbers}")


if __name__ == '__main__':
    tran_mlp_sklearn()
