import numpy as np

from data.augmentation.image import convolve


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kernels = np.random.randn(
            out_channels, in_channels, kernel_size, kernel_size)
        self.bias = np.random.randn(out_channels)
        self.input = None
        self.output = None
        self.grad = None
        self.grad_kernels = None
        self.grad_bias = None

    def forward(self, input):
        self.input = input
        self.output = np.zeros(
            (self.out_channels, (input.shape[0] - self.kernel_size + 1) // self.stride, (input.shape[1] - self.kernel_size + 1) // self.stride))
        for i in range(self.out_channels):
            self.output[i] = convolve(
                input, self.kernels[i], stride=self.stride, padding=self.padding) + self.bias[i]
        return self.output

    def backward(self, grad):
        self.grad = np.zeros_like(self.input)
        self.grad_kernels = np.zeros_like(self.kernels)
        self.grad_bias = np.zeros_like(self.bias)
        for i in range(self.out_channels):
            self.grad_kernels[i] = convolve(
                self.input, grad[i], stride=self.stride, padding=self.padding)
            self.grad_bias[i] = np.sum(grad[i])
            self.grad += convolve(grad[i], self.kernels[i][::-1, ::-1])
        return self.grad
