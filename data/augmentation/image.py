import numpy as np

# perform image augmentation for training


def flip(image):
    return image[:, ::-1]


def rotate(image, angle):
    return np.rot90(image, k=angle//90)


def scale(image, scale):
    h, w = image.shape
    new_h, new_w = int(h*scale), int(w*scale)
    new_image = np.zeros((new_h, new_w))
    for i in range(new_h):
        for j in range(new_w):
            new_image[i, j] = image[int(i/scale), int(j/scale)]
    return new_image


def translate(image, dx, dy):
    h, w = image.shape
    new_image = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            if i+dy >= 0 and i+dy < h and j+dx >= 0 and j+dx < w:
                new_image[i+dy, j+dx] = image[i, j]
    return new_image


def add_noise(image, noise):
    return image + np.random.randn(*image.shape)*noise


def add_blur(image, blur):
    kernel = np.ones((blur, blur))
    kernel /= kernel.size
    return convolve(image, kernel)


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
