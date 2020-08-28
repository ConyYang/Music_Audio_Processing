import math


def sigmoid_activation(h):
    y = 1./(1. + math.exp(-h))
    return y


def neuron_cal(input, weight):
    h = 0
    for x, w in zip(input, weight):
        h += x*w
    return sigmoid_activation(h)


if __name__ == "__main__":
    input = [.5, .3, .2]
    weight = [.4, .7, .2]
    output = neuron_cal(input, weight)
    print(output)