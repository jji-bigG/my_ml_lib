import numpy as np


# vectorized version of convolve
def convolve(image, kernel, stride=1, padding=True):
    if padding:
        h = (kernel.shape[0]-1)//2
        w = (kernel.shape[1]-1)//2
        image = np.pad(image, ((h, h), (w, w)), 'constant')
    output = np.zeros(
        (image.shape[0] - kernel.shape[0] + 1, image.shape[1] - kernel.shape[1] + 1))
    for x in range(output.shape[1]):
        for y in range(output.shape[0]):
            output[y, x] = np.sum(
                image[y:y + kernel.shape[0], x:x + kernel.shape[1]] * kernel)
    return output[::stride, ::stride]


def maxpool(image, pool_size):
    output = np.zeros(
        (image.shape[0] // pool_size, image.shape[1] // pool_size))
    for x in range(output.shape[1]):
        for y in range(output.shape[0]):
            output[y, x] = np.max(
                image[y * pool_size:y * pool_size + pool_size, x * pool_size:x * pool_size + pool_size])
    return output


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
            self.grad += convolve(grad[i], np.rot90(
                self.kernels[i], 2), stride=self.stride, padding=self.padding)
        return self.grad
