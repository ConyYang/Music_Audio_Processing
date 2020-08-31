"""
Here are the steps we need to follow:
1. Save activations and derivatives
2. Implement backpropagation
3. Implement gradient descent
4. Implement train
5. Train the network with dummy dataset
6. make predictions
"""
import numpy as np


class MLP:
    """
    A multilayer perceptron class
    """
    def __init__(self, num_inputs =3, num_hidden=[3, 3], num_outputs=2):
        """ Constructor for the MLP.
        :param num_inputs:  Number of inputs
        :param num_hidden: A list of integers for the hidden layer
        :param num_outputs: Number of outputs
        """
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        # Create a generic representation of the layers
        layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]

        # init random weights
        self.weights = []
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1])
            self.weights.append(w)

        # print(self.weights)

        self.activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            self.activations.append(a)

        self.derivatives = []
        for i in range(len(layers)-1):
            d = np.zeros((layers[i], layers[i+1]))
            self.derivatives.append(d)

    def forward_propagate(self, inputs):
        """
        Computes forward propagation for the network based on input signals
        :param inputs: ndarray: input signals
        :return: ndarray: Output values
        """
        # the input layer activation is just input itself
        activations = inputs
        self.activations[0] = inputs

        for i, w in enumerate(self.weights):
            # calculate matrix multiplication between previous activation and weight matrix
            net_inputs = np.dot(activations, w)
            # apply sigmoid activation function
            activations = self._sigmoid(net_inputs)
            self.activations[i+1] = activations
        # return output layer activations
        return activations

    def backward_propagate(self, loss, verbose):
        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i+1]
            delta = loss*self._sigmoid_derivative(activations)  # ndarray ([0.1, 0.2]) --> ndarray([[0.1, 0.2]])
            delta_reshaped = delta.reshape(delta.shape[0], -1).T
            current_activation = self.activations[i] # ndarray ([0.1, 0.2]) --> ndarray([[0.1],[0.2]])
            current_activations_reshaped = current_activation.reshape(current_activation.shape[0], -1)

            self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)
            loss = np.dot(delta, self.weights[i].T)
            if verbose:
                print("Derivatives for W{}: {}".format(i, self.derivatives[i]))
        return loss


    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)
    def _sigmoid(self, x):
        return 1/(1+np.exp(-x))


if __name__ == '__main__':
    # create an MLP
    mlp = MLP(2, [5], 1)
    # create input
    a = np.array([0.1, 0.2])
    target = np.array([0.3])
    # forward propagation
    output = mlp.forward_propagate(a)
    # calculate loss
    loss = target-output
    # backward propagation
    mlp.backward_propagate(loss=loss, verbose=True)
    # Print output
    print('The output of MLP is {}'.format(output))