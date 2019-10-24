import numpy as np


class Perceptron:
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
        self.weights = np.random.rand(np.shape(inputs)[1], 10) * 0.1 - 0.05

    def train(self, learning_rate, iterations):
        accuracy = np.zeros(iterations)
        for e in range(iterations):
            positives = 0
            negatives = 0
            print("Start of Epoch: ", e)
            print("Learning Rate: ", learning_rate)

            for k, inputs in enumerate(self.inputs):
                activations = np.dot(inputs, self.weights)
                result = np.where(activations == np.amax(activations))

                targets = np.zeros(10, dtype=float)
                targets[result[0]] = 1.

                if result[0] != self.targets[k]:
                    activations = np.where(activations > 0, 1, 0)
                    reshape_inputs = np.reshape(inputs, (-1, 1))
                    reshape_activations = np.reshape(activations-targets, (1, -1))
                    self.weights -= learning_rate*np.dot(reshape_inputs, reshape_activations)
                    negatives += 1
                else:
                    positives += 1
            accuracy[e] = positives / (positives+negatives)
            print("End of Epoch: ", e)
            print("Total: ", positives+negatives)
            print("Positive Outcomes: ", positives)
            print("Negative Outcomes: ", negatives)
            print("Accuracy: ", accuracy[e])

    def test(self):
        positive = 0
        negative = 0

        for k, inputs in enumerate(self.inputs):
            activations = np.dot(inputs, self.weights)
            result = np.where(activations == np.amax(activations))
            if result[0] == self.targets[k]:
                positive += 1
            else:
                negative += 1

        accuracy = positive / (positive + negative)
        print('Total: ', positive + negative)
        print('Positive Outcomes: ', positive)
        print('Negative Outcomes: ', negative)
        print('Accuracy: ', accuracy)


if __name__ == "__main__":
    csv_file = "mnist_train.csv"

    print("Loading csv file . . . ")
    data = np.loadtxt(csv_file, delimiter=',', max_rows=200)
    print("Complete")

    targets = np.asarray(data[:, :1], dtype='float')
    inputs = np.asarray(data[:, 1:], dtype='float')
    inputs = inputs / 255.0
    inputs = np.concatenate((np.ones((np.shape(inputs)[0], 1)), inputs), axis=1)

    pcn = Perceptron(inputs, targets)
    # pcn.test()
    pcn.train(0.01, 10)
