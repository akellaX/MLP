import numpy as np


class Layer:
    x = None
    z = None
    eps = 1e-9

    def Loss(self, y_true, y_calculated):
        eps = np.zeros(y_calculated.shape) + self.eps
        y = np.maximum(eps, np.minimum(1 - eps, y_true))
        return -np.sum((y_calculated * np.log(y) + (1 - y_calculated) * np.log(1 - y))) / y_calculated.shape[1] / y_calculated.shape[0]

    def __init__(self, in_size, out_size):
        self.W = np.random.randn(out_size, in_size)*0.01
        self.b = np.zeros((out_size, 1))
        self.grad_W = np.zeros((out_size, in_size))
        self.grad_b = np.zeros((out_size, 1))

    def getZ(self, date):
        Z = np.dot(self.W, date) + self.b
        return Z

    def sigmoida(self, Z):
        return 1 / (1+np.exp(-Z))

    def def_sigmoida(self, Z):
        return self.sigmoida(Z)*(1-self.sigmoida(Z))

    def oneHotEncode(self, arr, classes):
        if len(arr.shape) > 1:
            arr = arr.reshape(len(arr), )
        b = np.zeros((len(arr), classes), dtype=np.int)
        b[np.arange(len(arr)), arr] = 1
        return b

    def goForward(self, date):
        self.Z = self.getZ(date)
        self.x = date
        A = self.sigmoida(self.Z)
        return A

    def goBackward(self, grad):
        self.grad_Z = self.def_sigmoida(self.Z) * grad
        self.grad_b = np.sum(self.grad_Z, axis=-1, keepdims=True)
        self.grad_W = np.dot(self.grad_Z, np.transpose(self.x))
        self.grad_x = np.dot(np.transpose(self.W), self.grad_Z)
        return self.grad_x

    def accuracy(self, y_true, y_calculated):
        sum = 0;
        # print(y_true)
        # print(y_calculated)
        for i in range(len(y_true)):
            if (np.array_equal(y_true[i], y_calculated[i])):
                sum+=1
        return sum/len(y_true)
