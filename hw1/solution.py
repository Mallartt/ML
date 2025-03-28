import numpy as np


def product_of_diagonal_elements_vectorized(matrix: np.array):
    diag = np.diag(matrix).copy()
    diag[diag == 0] = 1
    return diag.prod()


def are_equal_multisets_vectorized(x: np.array, y: np.array):
    x.sort()
    y.sort()
    if len(x) != len(y):
        return False
    return (x == y).all()


def max_before_zero_vectorized(x: np.array):
    zeros = np.where(x == 0)[0]
    after_zeros_index = zeros + 1
    after_zeros_index = after_zeros_index[after_zeros_index < len(x)]
    after_zeros = x[after_zeros_index]
    return np.max(after_zeros)


def add_weighted_channels_vectorized(image: np.array):
    height, width, numChannels = image.shape
    gray_image = np.dot(image[:, :, :3], np.array([0.299, 0.587, 0.114]))
    return gray_image


def run_length_encoding_vectorized(x: np.array):
    diff_x = np.diff(x)
    diff_index = np.concatenate([[0], np.where(diff_x != 0)[0] + 1])
    uniques = x[diff_index]
    diff_index = np.concatenate([diff_index, [len(x)]])
    length = np.diff(diff_index)
    return (uniques, length)
