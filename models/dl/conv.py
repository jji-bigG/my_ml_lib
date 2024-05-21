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
