import numpy as np
# TODO Fix weights not updating properly

class Perceptron:
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
        self.weights = np.random.rand(np.shape(inputs)[1], 10) * 0.1 - 0.05

    def train(self, learning_rate, iterations):
        accuracy = np.zeros(iterations)

        # Start epoch
        for e in range(iterations):
            positives = 0
            negatives = 0
            print("Start of Epoch: ", e)
            print("Learning Rate: ", learning_rate)

            # Start enumerating through data-set
            for k, inputs in enumerate(self.inputs):
                activations = np.dot(inputs, self.weights)
                result = np.where(activations == np.max(activations))
                # print(result[0])
                # print(self.targets[k])
                # print("End")

                t_targets = np.zeros(10, dtype=float)
                t_targets[result[0]] = 1.

                # if results don't match the target, update weights
                if result[0] == self.targets[k]:
                    positives += 1
                else:
                    y_activations = np.where(activations > 0, 1, 0)
                    reshape_inputs = np.reshape(inputs, (-1, 1))
                    reshape_activations = np.reshape(t_targets - y_activations, (1, -1))
                    self.weights = self.weights + learning_rate * np.dot(reshape_inputs, reshape_activations)
                    negatives += 1

            # Compute accuracy and output results
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
    data = np.loadtxt(csv_file, delimiter=',', max_rows=500)
    print("Complete")

    targets = np.asarray(data[:, :1], dtype='float')
    inputs = np.asarray(data[:, 1:], dtype='float')
    inputs = inputs / 255.0
    inputs = np.concatenate((np.ones((np.shape(inputs)[0], 1)), inputs), axis=1)

    pcn = Perceptron(inputs, targets)

    pcn.train(0.01, 70)
    # pcn.test()
