import numpy as np
import Layer

class MLP:
    speed = 0.1
    eps = 1e-2

    def __init__(self, in_size, layers):
        self.layers = [Layer.Layer(in_size,layers[0])]
        for i in range(len(layers)-1):
            self.layers.append(Layer.Layer(layers[i],layers[i+1]))

    def forward(self, arr):
        a = self.layers[0].goForward(arr)
        for i in range(len(arr)-1):
            a = self.layers[i+1].goForward(a)
        return a

    def backward(self, grad):
        for i in reversed(self.layers):
            grad = i.goBackward(grad)

    def change_weights(self):
        is_updated = False
        for i in self.layers:
            if (np.linalg.norm(i.grad_W) / (i.grad_W.shape[0] * i.grad_W.shape[1]) > self.eps):
                is_updated = True
            i.W = i.W - i.grad_W * self.speed
            i.b = i.b - i.grad_b * self.speed
        return is_updated