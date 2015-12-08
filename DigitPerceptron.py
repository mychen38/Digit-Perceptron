import logging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import time

from evaluation import calc_accuracy, confusion_matrix


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = './data'


class DigitPerceptron:

    def __init__(self):
        logger.info('2/1+t')
        self.epochs = 80
        self.alpha = 1.0
        self.num_classes = 10
        self.col = 28
        self.row = 28
        self.bias = False
        self.random_weights = True

        if (self.random_weights):
            self.feature_weight_vectors = np.random.uniform(-2, 2, (self.num_classes, self.row, self.col))
        else:
            self.feature_weight_vectors = np.zeros((self.num_classes, self.row, self.col))

        if (self.bias):
            self.bias_val = 1
            self.bias_weight = np.zeros(self.num_classes)

    def decay_alpha(self, t):
        self.alpha = 2 / (1 + t)

    def train_decision(self, model):
        dot_products = [0]*self.num_classes
        for i in range(self.num_classes):
            for y in range(self.row):
                for x in range(self.col):
                    dot_products[i] += model[y][x] * self.feature_weight_vectors[i][y][x]
            if self.bias:
                dot_products[i] += self.bias_weight[i] * self.bias_val
        return np.argmax(dot_products)

    def update_weights(self, decision, label, model):
        for y in range(self.row):
            for x in range(self.col):
                self.feature_weight_vectors[decision][y][x] -= self.alpha * model[y][x]
                self.feature_weight_vectors[label][y][x] += self.alpha * model[y][x]
        if self.bias:
            self.bias_weight[decision] -= self.alpha * self.bias_val
            self.bias_weight[label] += self.alpha * self.bias_val

    def sorted_train(self):
        training_label_path = DATA_DIR + '/traininglabels'
        training_images_path = DATA_DIR + '/trainingimages'

        labels = []

        with open(training_label_path) as f:
            for line in f:
                labels.append(int(line))

        sorted_label_index = np.argsort(labels).tolist()
        num_images = len(labels)

        training_images = [[None for _ in range(self.row)] for _ in range(num_images)]
        with open(training_images_path) as f:
            for n in range(num_images):
                for y in range(self.row):
                    training_images[n][y] = list(f.readline().rstrip('\n'))

        start_time = time.time()
        for t in range(1, self.epochs + 1):
            self.decay_alpha(t)
            total = 0
            correct = 0
            for n in sorted_label_index:
                model = np.zeros((self.row,self.col))
                total += 1
                for y in range(self.row):
                    for x in range(self.col):
                        if training_images[n][y][x] in ['+', '#']:
                            model[y][x] = 1
                decision = self.train_decision(model)
                if decision != labels[n]:
                    self.update_weights(decision, labels[n], model)
                else:
                    correct += 1
            accuracy = 100*correct/total
            logger.info('Correct: {0}, Total: {1}'.format(correct,total))
            logger.info('Accuracy for epoch {0} is {1:.2f}%'.format(t, accuracy))
        logger.info('Finished training in {0:.2f} seconds'.format(time.time() - start_time))

    def train(self):
        training_label_path = DATA_DIR + '/traininglabels'
        training_images_path = DATA_DIR + '/trainingimages'

        labels = []

        with open(training_label_path) as f:
            for line in f:
                labels.append(int(line))

        num_images = len(labels)

        training_images = [[None for _ in range(self.row)] for _ in range(num_images)]
        with open(training_images_path) as f:
            for n in range(num_images):
                for y in range(self.row):
                    training_images[n][y] = list(f.readline().rstrip('\n'))

        start_time = time.time()

        for t in range(1, self.epochs + 1):
            self.decay_alpha(t)
            total = 0
            correct = 0
            for n in range(len(labels)):
                model = np.zeros((self.row,self.col))
                total += 1
                for y in range(self.row):
                    for x in range(self.col):
                        if training_images[n][y][x] in ['+', '#']:
                            model[y][x] = 1
                decision = self.train_decision(model)
                if decision != labels[n]:
                    self.update_weights(decision, labels[n], model)
                else:
                    correct += 1
            accuracy = 100*correct/total
            logger.info('Correct: {0}, Total: {1}'.format(correct,total))
            logger.info('Accuracy for epoch {0} is {1:.2f}%'.format(t, accuracy))
        logger.info('Finished training in {0:.2f} seconds'.format(time.time() - start_time))
        if self.bias:
            logger.info(self.bias_weight)

    def predict(self, info=True):
        test_label_path = DATA_DIR + '/testlabels'
        test_images_path = DATA_DIR + '/testimages'

        correct_labels = []

        with open(test_label_path) as f:
            for line in f:
                correct_labels.append(int(line))

        num_images = len(correct_labels)

        test_images = [[None for _ in range(self.row)] for _ in range(num_images)]
        with open(test_images_path) as f:
            for n in range(num_images):
                for y in range(self.row):
                    test_images[n][y] = list(f.readline().rstrip('\n'))

        predicted_labels = []

        for n in range(num_images):
            model = np.zeros((self.row,self.col))
            for y in range(self.row):
                for x in range(self.col):
                    if test_images[n][y][x] in ['+', '#']:
                        model[y][x] = 1
            decision = self.train_decision(model)
            predicted_labels.append(decision)

        truths = np.array(correct_labels)
        predictions = np.array(predicted_labels)
        accuracy = calc_accuracy(truths, predictions)
        logger.info('NB model is {0:.2f}% accurate on the digit data'.format(accuracy))

        if info:
            confm = confusion_matrix(truths, predictions, self.num_classes)
            class_accuracies = [confm[n][n] for n in range(self.num_classes)]
            # Class accuracies
            for n, x in enumerate(class_accuracies):
                logger.info('Class {0} has an accuracy of {1:.2f}%'.format(n, 100 * x))

            # Confusion matrixx
            plt.figure()
            plt.imshow(confm, cmap=plt.get_cmap('Greens'), interpolation='nearest')
            plt.title('Confusion Matrix')
            plt.xticks(np.arange(self.num_classes))
            plt.yticks(np.arange(self.num_classes))
            plt.xlabel('Predictions')
            plt.ylabel('Truths')

            for i in range(self.num_classes):
                hf = plt.figure()
                ha = hf.gca(projection = '3d')

                X, Y = np.meshgrid(range(self.col), range(self.row))
                Y.reverse()
                ha.plot_surface(X, Y, self.feature_weight_vectors[i], rstride=1, cstride=1,
                                linewidth=0, cmap=cm.coolwarm, antialiased = False)
                ha.set_xlabel('X')
                ha.set_ylabel('Y')
                ha.set_zlabel('weigh')
            plt.show()



def main():
    dpt = DigitPerceptron()
    dpt.train()
    dpt.predict()


if __name__ == '__main__':
    main()
