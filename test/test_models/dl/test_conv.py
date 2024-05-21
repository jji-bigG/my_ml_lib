import unittest
import numpy as np

from models.dl.conv import Conv2D, convolve, maxpool

# Assuming the Conv2D class and helper functions convolve and maxpool are defined above


class TestConv2D(unittest.TestCase):
    def setUp(self):
        self.in_channels = 1
        self.out_channels = 1
        self.kernel_size = 3
        self.stride = 1
        self.padding = True
        self.conv = Conv2D(self.in_channels, self.out_channels,
                           self.kernel_size, self.stride, self.padding)
        self.input = np.random.randn(1, 5, 5)  # Single-channel 5x5 input
        # Same shape as input for gradient
        self.grad = np.random.randn(1, 5, 5)

    def test_convolve_shape(self):
        kernel = np.random.randn(3, 3)
        output = convolve(self.input[0], kernel, stride=1, padding=True)
        # Since padding is True and stride is 1
        expected_output_shape = (self.input.shape[1], self.input.shape[2])
        self.assertEqual(output.shape, expected_output_shape)

    def test_convolve_computation(self):
        input = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        expected_output = np.array([[0, 0, 0], [12, 0, -12], [0, 0, 0]])
        output = convolve(input, kernel, stride=1, padding=True)
        np.testing.assert_almost_equal(output, expected_output)

    def test_maxpool_shape(self):
        output = maxpool(self.input[0], pool_size=2)
        expected_output_shape = (
            self.input.shape[1] // 2, self.input.shape[2] // 2)
        self.assertEqual(output.shape, expected_output_shape)

    def test_maxpool_computation(self):
        input = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [
                         9, 10, 11, 12], [13, 14, 15, 16]])
        expected_output = np.array([[6, 8], [14, 16]])
        output = maxpool(input, pool_size=2)
        np.testing.assert_almost_equal(output, expected_output)

    def test_forward_shape(self):
        output = self.conv.forward(self.input)
        expected_output_shape = (
            self.out_channels,
            (self.input.shape[1] + 2 * ((self.kernel_size -
             1) // 2) - self.kernel_size) // self.stride + 1,
            (self.input.shape[2] + 2 * ((self.kernel_size -
             1) // 2) - self.kernel_size) // self.stride + 1
        )
        self.assertEqual(output.shape, expected_output_shape)

    def test_forward_computation(self):
        # Set kernels to ones for predictable output
        self.conv.kernels = np.ones_like(self.conv.kernels)
        self.conv.bias = np.zeros_like(self.conv.bias)  # Set biases to zero
        output = self.conv.forward(self.input)
        expected_output = convolve(
            self.input[0], self.conv.kernels[0][0], stride=self.stride, padding=self.padding)
        np.testing.assert_almost_equal(output[0], expected_output)

    def test_backward_shape(self):
        self.conv.forward(self.input)
        grad_input = self.conv.backward(self.grad)
        self.assertEqual(grad_input.shape, self.input.shape)

    def test_backward_computation(self):
        # Set kernels to ones for predictable output
        self.conv.kernels = np.ones_like(self.conv.kernels)
        self.conv.bias = np.zeros_like(self.conv.bias)  # Set biases to zero
        self.conv.forward(self.input)
        grad_input = self.conv.backward(self.grad)

        # Check grad_kernels computation
        expected_grad_kernels = np.array([
            convolve(self.input[0], self.grad[0],
                     stride=self.stride, padding=self.padding)
            for _ in range(self.out_channels)
        ])
        np.testing.assert_almost_equal(
            self.conv.grad_kernels[0], expected_grad_kernels[0])

        # Check grad_bias computation
        expected_grad_bias = np.sum(self.grad[0])
        self.assertAlmostEqual(self.conv.grad_bias[0], expected_grad_bias)

        # Check grad computation
        expected_grad_input = convolve(self.grad[0], np.rot90(np.ones(
            (self.kernel_size, self.kernel_size)), 2), stride=self.stride, padding=self.padding)
        np.testing.assert_almost_equal(grad_input[0], expected_grad_input)


if __name__ == '__main__':
    unittest.main()
