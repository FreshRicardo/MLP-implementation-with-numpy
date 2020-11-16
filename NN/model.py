import numpy as np

class Model:
    def __init__(self, learning_rate):
        """
        Initialize a deep model.

        Parameters
        ----------
        learning_rate: int
            The tuning parameter in an optimization algorithm.
        """
        self.learning_rate = learning_rate
        self.layers = []

    def add(self, layer):
        """
        Add a single layer to the network.
        """
        self.layers.append(layer)

    def forward(self, input):
        """
        A forward pass of a deep model.

        Parameters
        ----------
        input: np.array
            2d array, the input data feeded to the network.

        Returns
        -------
        list
            the output of each layer.
        """
        xs = [input]
        x = input
        for l in self.layers:
            xs.append(l.forward(x))
            x = xs[-1]
        return xs

    def softmax(self, x):
        """
        Softmax function

        Parameters
        ----------
        x: np.array
            2d array, the input of softmax.

        Returns
        -------
        np.array
            the result of softmax calculation.
        """
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def softmax_cross_entropy_with_logits(self, labels, logits):
        """
        Cross entropy loss with softmax.

        Parameters
        ----------
        labels: np.array
            2d array, labels of input data.
        logits: np.array
            2d array, logtis predicted by the network.

        Returns
        -------
        float
            the cross entropy loss.
        """
        p = self.softmax(logits)
        loss = -np.mean(labels * np.log(p + 1e-15))
        return loss

    def gradient_loss(self, labels, logits):
        """
        Derivative of cross entropy loss with softmax.

        Parameters
        ----------
        labels: np.array
            2d array, labels of input data.
        logits: np.array
            2d array, logtis predicted by the network.

        Returns
        -------
        np.array
            derivative of cross entropy loss with Softmax.
        """
        p = self.softmax(logits)
        return p - labels

    def train(self, x, y):
        """
        Train process of the network.

        Parameters
        ----------
        x: np.array
            2d array, input features.
        y: np.array
            2d array, input labels.

        Returns
        -------
        float
            loss divided by number of samples.
        """
        xs = self.forward(x)
        y_pred = xs[-1]
        loss = self.softmax_cross_entropy_with_logits(y, y_pred)
        sigma = self.gradient_loss(y, y_pred)

        w_next = None
        for l_idx in range(len(self.layers))[::-1]:
            layer = self.layers[l_idx]
            x_layer = xs[l_idx]

            w_save = layer.w.copy()
            sigma = layer.backprop(x_layer, sigma, w_next, self.learning_rate)

            w_next = w_save

        return loss

    def batch_generator(self, x, y, batch_size):
        """
        Split data into batches for mini-batch gradient descent.

        Parameters
        ----------
        x: np.array
            2d array, input features.
        y: np.array
            2d array, input labels.

        Returns
        -------
        generator
            a generator of batches.
        """
        for i in range(0, len(x), batch_size):
            yield x[i:i + batch_size], y[i:i + batch_size]

    def fit(self, x, y, batch_size):
        """
        Fit the network.

        Parameters
        ----------
        x: np.array
            2d array, input features.
        y: np.array
            2d array, input labels.

        Returns
        -------
        float
            mean loss of batches.
        """
        losses = []
        for train_x, train_y in self.batch_generator(x, y, batch_size):
            batch_loss = self.train(train_x, train_y)
            losses.append(batch_loss)
        return np.mean(losses)

    def predict(self, input):
        """
        Compute predictions.

        Parameters
        ----------
        x: np.array
            2d array, input features.
        y: np.array
            2d array, input labels.

        Returns
        -------
        np.array
            indices of the largest logit for each sample.
        """
        y_pred = self.forward(input)[-1]
        return y_pred.argmax(axis=-1)