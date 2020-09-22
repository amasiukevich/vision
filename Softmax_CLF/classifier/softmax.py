import numpy as np


class Softmax():

    def __init__(self, learning_rate=0.01, regul_lambda=1, max_epochs=20, sat_loss=1000, batch_size=100):

        """
        :param learning_rate: learning rate for gradient descend - single value for simplicity
        :param regul_lambda:  lambda coefficient for regularizations
        """
        self.learning_rate = learning_rate
        self.regul_lambda = regul_lambda
        self.satisfying_loss = sat_loss
        self.minibatch_size = batch_size
        self.max_epochs = max_epochs


    def fit(self, X, y):

        """
        :param X: numpy array of features and associated values
        :param y: numpy array of true labels
        :return:  "void"
        """
        assert len(X) == len(y)
        # self.minibatch_size = int(np.sqrt(len(X)))

        ### Add here a 1 to the end of every training_example
        n_elements = int(np.prod(np.array(X.shape)) / len(X))
        partial_train = np.reshape(X, newshape=(len(X), n_elements))
        self.train_X = np.array([np.append(el, 1) for el in partial_train])

        self.train_y = y

        self.num_classes = len(set(self.train_y))

        # Initialize the weights
        self.weights = self.generate_weights()

        epochs = 0

        best_loss = np.inf
        epochs_without_improvement = 0
        while epochs < self.max_epochs and epochs_without_improvement < 1:

            shuffled_training_X, shuffled_training_y = self.shuffle_trainings()
            minibatches_per_one_epoch = self.prepare_minibatches(shuffled_training_X, shuffled_training_y)

            avg_loss = self.gradient_descent(minibatches_per_one_epoch)
            print(f"\nEpoch {epochs + 1}\nAvg loss in this epoch:\t{avg_loss}")


            if avg_loss < best_loss:
                best_loss = avg_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            epochs += 1

        print("Done!")

        # self.gradient_descent()

    def generate_weights(self):

        """
        A method to generate the initial weights for the algorithm
        :return:
        """

        weights = np.random.uniform(0, 1 / self.train_X.shape[1], size=(self.train_X.shape[1], self.num_classes))
        return weights


    def shuffle_trainings(self):

        """
        A simple method to shuffle training arrays (X, y) in the same order
        :return: shuffled features and shuffled labels
        """
        # TODO: write tests for this function

        index = np.random.permutation(len(self.train_X))
        shuffled_X = self.train_X[index]
        shuffled_y = self.train_y[index]

        return shuffled_X, shuffled_y

    def prepare_minibatches(self, shuffled_X, shuffled_y):

        """
        A method to divide the dataset into multiple minibatches to perform batch gradient descend
        :param shuffled_X: - the shuffled copy of training examples set
        :param shuffled_y: - the shuffled copy of testing labels set
        :return:
        """

        minibatches = []
        for i in range(len(shuffled_X) // self.minibatch_size):
            minibatches.append(
                (shuffled_X[i * self.minibatch_size: (i + 1) * self.minibatch_size],
                shuffled_y[i * self.minibatch_size: (i + 1) * self.minibatch_size])
            )

        ## The last one
        if len(shuffled_X) % self.minibatch_size != 0:
            minibatches.append((shuffled_X[(i + 1) * self.minibatch_size:], shuffled_y[(i + 1) * self.minibatch_size:]))

        assert len(minibatches[-1][0]) <= len(minibatches[1][0])
        assert all([len(minibatches[i][0]) == len(minibatches[i][1]) for i in range(len(minibatches))])
        return minibatches


    def gradient_descent(self, minibatches):

        losses = np.array([])
        for minibatch_X, minibatch_y in minibatches:

            mul_results = minibatch_X.dot(self.weights)
            softmaxed = self.calc_softmax(mul_results)

            loss, dprediction = self.calc_log_loss(softmaxed, minibatch_y)
            losses = np.append(losses, [loss])

            # Getting the improvements
            improvements = self.get_full_gradient(dprediction)
            self.weights -= self.learning_rate * improvements

        avg_loss = np.sum(losses) / np.sum([len(i[0]) for i in minibatches])
        # avg_loss = np.mean(losses)

        return avg_loss


    def calc_softmax(self, mul_results):
        """
        :param mul_results: result of matrix multiplication of minibatch and weights
                of shape (minibatch_size, num_classes)
        :return: softmaxed probabilities
                of shape (minibatch_size, num_classes)
        """

        # Mathematical trick to avoid big numbers
        mul_results -= np.max(mul_results)

        numerator = np.exp(mul_results)
        denominator_values = np.sum(np.exp(mul_results), axis=1)
        denominator = np.reshape(np.repeat(denominator_values, numerator.shape[1]), numerator.shape)

        # breakpoint()
        softmaxed = numerator / denominator

        return softmaxed

    def calc_log_loss(self, softmaxed, target_index):

        """
        Computes log loss and the gradient of the loss function by softmaxed probabilities
        :param softmaxed: softmaxed probabilities
                            of shape (minibatch_size
        :param target_index: the only true values of the given example
        :return: log loss and gradient by softmaxed probs
        """

        # breakpoint()
        logarithmed_probs = np.log(softmaxed)

        rearranged_targets = np.zeros(shape=(len(target_index), self.num_classes))
        rearranged_targets[np.arange(len(target_index)), target_index.T] = 1

        loss_matr = rearranged_targets * logarithmed_probs

        ### Loss with the regularization
        loss = (-1) * np.sum(loss_matr) + self.regul_lambda * (np.linalg.norm(self.weights) ** 2)

        dprediction = softmaxed - rearranged_targets
        return loss, dprediction

    def choose_rand_minibatch(self):

        random_indexes = np.random.randint(0, len(self.train_X), size=self.minibatch_size)
        return self.train_X[random_indexes], self.train_y[random_indexes]


    def get_full_gradient(self, dprediction):

        ### dummy loop implementation
        updates_by_example = []
        for el in dprediction:
            tiled_dprediction = np.reshape(np.tile(el, self.weights.shape[0]),
                                           newshape=(self.weights.shape[0], self.num_classes))

            improvements_one = tiled_dprediction * self.weights
            updates_by_example.append(improvements_one)

        # TODO: Loopless implementation

        return np.mean(updates_by_example, axis=0)

    def predict(self, X_test):

        n_elements = int(np.prod(np.array(X_test.shape)) / len(X_test))
        partial_test = np.reshape(X_test, newshape=(len(X_test), n_elements))
        X_test = np.array([np.append(el, 1) for el in partial_test])

        mult_results = X_test.dot(self.weights)

        softmaxed = self.calc_softmax(mult_results)

        predictions = []

        for el in softmaxed:

            index = np.where(el == max(el))[0][0]
            predictions.append(index)

        # TODO: Loopless implementation

        breakpoint()
        return predictions