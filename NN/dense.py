import numpy as np

class Dense:
    def __init__(self, input_size, output_size, activation='sigmoid', kernel_initializer='MSRA',
                 bias_initializer='zeros'):
        """
        Initialize a dense layer.

        Parameters
        ----------
        input_size : int
            The input size of the layer
        output_size : int
            The output size of the layer
        activation : string
            The activation function for neurons
        kernel_initializer: string
            Initializer for the kernel weights matrix.
        bias_initializer: string
             Initializer for the bias vector.
        """
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.initialize(input_size, output_size)

    def initialize(self, input_size, output_size):
        """
        Initialize the kernel weights matrix and the bias vector.
        Xavier and MSRA initializaitons are provided.
        """
        if self.kernel_initializer == 'Xavier':
            var = 2. / (input_size + output_size)
            bound = np.sqrt(3.0 * var)
            self.w = np.random.uniform(-bound, bound, size=(input_size, output_size))

        if self.kernel_initializer == 'MSRA':
            s = np.sqrt(2 / (input_size + output_size))
            self.w = np.random.normal(loc=0.0, scale=s, size=(input_size, output_size))

        if self.bias_initializer == 'zeros':
            self.b = np.zeros(output_size)

    def forward(self, x):
        """
        Returns a forward result of a layer.
        """
        return self.activation_func(np.dot(x, self.w) + self.b)

    def activation_func(self, x):
        """
        Returns the activation result.
        Sigmoid and Relu are provided.
        """
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        if self.activation == 'relu':
            return np.maximum(x, 0)
        if self.activation == 'None':
            return x

    def grad_activation(self, x):
        """
        Returns the derivative of the activation function.
        """
        if self.activation == 'sigmoid':
            a = self.activation_func(x)
            return a * (1 - a)
        if self.activation == 'relu':
            x_ = x.copy()
            x_[x_ <= 0] = 0
            x_[x_ > 0] = 1
            return x_

    def backprop(self, x, sigma, w_next, learning_rate):
        """
        Back-propagation process of a layer.

        Parameters
        ----------
        x : np.array
            2d array, the input of the layer
        sigma : np.array
            2d array, the iteration term in the Back-propagation process.
        w_next : np.array
            2d array, the weights matrix of the next layer.
        learning_rate: int
            The tuning parameter in an optimization algorithm.

        Returns
        -------
        np.array
            the iteration term for next backprop calculation.
        """
        wx = np.dot(x, self.w) + self.b
        if w_next is not None:
            sigma = np.multiply(np.dot(sigma, w_next.T), self.grad_activation(wx))

        gradient_w = np.dot(x.T, sigma)
        gradient_b = sigma.sum(axis=0)

        self.w -= learning_rate * gradient_w / x.shape[0]
        self.b -= learning_rate * gradient_b / x.shape[0]

        return sigma