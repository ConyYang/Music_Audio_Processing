
"""
numpy.linspace
numpy.linspace(start, stop, num=50, endpoint=True,
retstep=False, dtype=None, axis=0）
num: int, optional; Number of samples to generate. Default: 50
endpoint: bool, optional; if true, stop is the last sample. if false, stop not include
"""
import numpy as np
import matplotlib.pyplot as plt

def funct_1():
    a = np.linspace(start=2.0, stop=3.0, num=5)
    print(a)
    b = np.linspace(start=2.0, stop=3.0, num=5, endpoint=False)
    print(b)

def graph_illustration():
    step = 8
    y1 = np.zeros(step)
    y2 = y1 + 0.5

    x1 = np.linspace(start=0.0, stop=10.0, num=step, endpoint=True)
    x2 = np.linspace(start=0.0, stop=10.0, num=step, endpoint=False)

    plt.plot(x1, y1, 'o')
    plt.plot(x2, y2, 'o')
    plt.ylim([-0.5, 1])
    plt.show()

print(range(10))