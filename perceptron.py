import numpy as np
import pylab as pl
from sklearn.metrics import confusion_matrix
# TODO Fix weights not updating properly

class Perceptron:


    def plot_training(self, epoch, accuracy, learning_rate):
        x = range(0, epoch+1)
        y = accuracy

        pl.ion()
        pl.figure()
        pl.plot(x, y)
        pl.xlabel('Epoch')
        y_label = 'Accuracy of Set - Learning Rate: %f' % learning_rate
        pl.ylabel(y_label)
        pl.title('Accuracy of Set at each Epoch')
        pl.show();

    def train(self, inputs, targets, weights, learning_rate, iterations):
        accuracy = []
        epoch = 0

        # Start epoch
        for e in range(iterations):
            positives = 0
            negatives = 0
            epoch = e
            print("Start of Epoch: ", epoch)
            print("Learning Rate: ", learning_rate)

            # Start enumerating through data-set
            for k, inputs_set in enumerate(inputs):
                activations = np.dot(inputs_set, weights)
                result = np.argmax(activations)

                # if results don't match the target, update weights
                if result == targets[k]:
                    positives += 1
                else:
                    t_targets = np.zeros(10, dtype='float')
                    t_targets[targets[k]] = 1.

                    activations = np.where(activations > 0, 1, 0)
                    weights = weights + learning_rate * np.transpose(np.dot(np.reshape((t_targets - activations), (-1, 1)), np.reshape(inputs_set, (1, -1))))
                    negatives += 1

            # Compute accuracy and output results
            accuracy.append(positives / (positives+negatives))
            print("End of Epoch: ", epoch)
            print("Total: ", positives+negatives)
            print("Positive Outcomes: ", positives)
            print("Negative Outcomes: ", negatives)
            print("Accuracy: ", accuracy[e], "\n")

            # if (accuracy[epoch] - accuracy[epoch-1]) < 0.01:
            #    self.plot_training(epoch, accuracy)
            #    break

        # self.plot_training(epoch, accuracy, learning_rate)
        return weights

    def test(self, inputs, targets, weights):
        true_positive = 0
        false_positive = 0

        expected = np.zeros(10, dtype='int')
        predicted = np.zeros(10, dtype='int')


        for k, inputs_set in enumerate(inputs):
            activations = np.dot(inputs_set, weights)

            result = np.argmax(activations)
            t_targets = np.zeros(10, dtype='int')
            t_targets[targets[k]] = 1

            expected = expected + t_targets
            predicted = predicted + np.where(activations > 0, 1, 0)
            if result == targets[k]:
                true_positive += 1
            else:
                false_positive += 1

        accuracy = true_positive / (true_positive + false_positive)
        print('Total: ', true_positive + false_positive)
        print('Positive Outcomes: ', true_positive)
        print('Negative Outcomes: ', false_positive)
        print('Accuracy: ', accuracy)

        cm = confusion_matrix(expected, predicted)
        print(cm)


if __name__ == "__main__":
    csv_file = "mnist_train.csv"

    print("Loading csv file . . . ")
    data = np.loadtxt(csv_file, delimiter=',', max_rows=5000)
    print("Complete")

    targets = np.asarray(data[:, :1], dtype='int')
    inputs = np.asarray(data[:, 1:], dtype='float')
    inputs = inputs / 255.0
    inputs = np.concatenate((np.ones((np.shape(inputs)[0], 1)), inputs), axis=1)
    weights = np.random.rand(np.shape(inputs)[1], 10) * 0.1 - 0.05

    pcn = Perceptron()
    pcn.test(inputs, targets, weights)

    weights = pcn.train(inputs, targets, weights, 0.1, 70)
    # pcn.test(inputs, targets, weights)
