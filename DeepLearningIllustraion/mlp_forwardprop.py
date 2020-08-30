import numpy as np


class MLP:
    def __init__(self, num_inputs =3, num_hidden=[3, 3], num_outputs=2):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]

        # init random weights
        self.weights = []
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1])
            self.weights.append(w)

        print(self.weights)

    def forward_propagate(self, inputs):
        activations = inputs
        for w in self.weights:
            # calculate net inputs
            net_inputs = np.dot(activations, w)
            # calculate the activations
            activations = self._sigmoid(net_inputs)
        return activations

    def _sigmoid(self, x):
        return 1/(1+np.exp(-x))


if __name__ == '__main__':
    # create an MLP
    mlp = MLP()
    # create input
    a = [1., 3., 4.]
    # forward propagation
    output = mlp.forward_propagate(a)
    # Print output
    print('The output of MLP is {}'.format(output))