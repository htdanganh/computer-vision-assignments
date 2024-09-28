#!/usr/bin/python3

from typing import Tuple

import numpy as np


def create_Gaussian_kernel_1D(ksize: int, sigma: int) -> np.ndarray:
    """Create a 1D Gaussian kernel using the specified filter size and standard deviation.

    The kernel should have:
    - shape (k,1)
    - mean = floor (ksize / 2)
    - values that sum to 1

    Args:
        ksize: length of kernel
        sigma: standard deviation of Gaussian distribution

    Returns:
        kernel: 1d column vector of shape (k,1)

    HINT:
    - You can evaluate the univariate Gaussian probability density function (pdf) at each
      of the 1d values on the kernel (think of a number line, with a peak at the center).
    - The goal is to discretize a 1d continuous distribution onto a vector.
    """

    sd = np.arange(-ksize // 2 + 1, ksize // 2 + 1)
    kernel = np.exp(-(sd ** 2) / (2 * sigma ** 2))
    kernel = (kernel / kernel.sum()).reshape(-1, 1)

    return kernel


def create_Gaussian_kernel_2D(cutoff_frequency: int) -> np.ndarray:
    """
    Create a 2D Gaussian kernel using the specified filter size, standard
    deviation and cutoff frequency.

    The kernel should have:
    - shape (k, k) where k = cutoff_frequency * 4 + 1
    - mean = floor(k / 2)
    - standard deviation = cutoff_frequency
    - values that sum to 1

    Args:
        cutoff_frequency: an int controlling how much low frequency to leave in
        the image.
    Returns:
        kernel: numpy nd-array of shape (k, k)

    HINT:
    - You can use create_Gaussian_kernel_1D() to complete this in one line of code.
    - The 2D Gaussian kernel here can be calculated as the outer product of two
      1D vectors. In other words, as the outer product of two vectors, each
      with values populated from evaluating the 1D Gaussian PDF at each 1d coordinate.
    - Alternatively, you can evaluate the multivariate Gaussian probability
      density function (pdf) at each of the 2d values on the kernel's grid.
    - The goal is to discretize a 2d continuous distribution onto a matrix.
    """

    k = cutoff_frequency * 4 + 1
    kernel = create_Gaussian_kernel_1D(k, cutoff_frequency)
    kernel = np.outer(kernel, kernel)
    
    return kernel


def separate_Gaussian_kernel_2D(kernel: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Separate a 2D kernel into two 1D kernels with Singular Value Decomposition(SVD).

    The two 1D kernels v and h should have:
    - shape (k, 1) where k is also the shape of the input 2D kernel
    - kernel = v * transpose(h), where kernel is the input 2D kernel

    Args:
        kernel: numpy nd-array of shape (k, k) representing a 2D Gaussian kernel that
        needs to be separated
    Returns:
        v: numpy nd-array of shape (k, 1)
        h: numpy nd-array of shape (k, 1)

    HINT:
    - You can use np.linalg.svd to take the SVD.
    - We encourage you to first check the separability of the 2D kernel, even though
      it might not be necessary for 2D Gaussian kernels.
    """

    u, s, vh = np.linalg.svd(kernel)
    v = u[:, 0].reshape(-1, 1) * np.sqrt(s[0])
    h = vh[0, :].reshape(-1, 1) * np.sqrt(s[0])

    return v, h


def my_conv2d_numpy(image: np.ndarray, filter: np.ndarray) -> np.ndarray:
    """Apply a single 2d filter to each channel of an image. Return the filtered image.

    Note: we are asking you to implement a very specific type of convolution.
      The implementation in torch.nn.Conv2d is much more general.

    Args:
        image: array of shape (m, n, c)
        filter: array of shape (k, j)
    Returns:
        filtered_image: array of shape (m, n, c), i.e. image shape should be preserved

    HINTS:
    - You may not use any libraries that do the work for you. Using numpy to
      work with matrices is fine and encouraged. Using OpenCV or similar to do
      the filtering for you is not allowed.
    - We encourage you to try implementing this naively first, just be aware
      that it may take an absurdly long time to run. You will need to get a
      function that takes a reasonable amount of time to run so that the TAs
      can verify your code works.
    - If you need to apply padding to the image, only use the zero-padding
      method. You need to compute how much padding is required, if any.
    - "Stride" should be set to 1 in your implementation.
    - You can implement either "cross-correlation" or "convolution", and the result
      will be identical, since we will only test with symmetric filters.
    """

    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1

    m, n, c = image.shape
    k, j = filter.shape

    h_pad = j // 2
    v_pad = k // 2
    pad = ((v_pad, v_pad), (h_pad, h_pad))

    filtered_image = np.zeros_like(image)

    for Z in range(c):
        ch = image[:, :, Z]
        pad_ch = np.pad(ch, pad)

        filtered_channel = np.zeros_like(ch)
        filtered_pad_ch = np.zeros_like(pad_ch)

        for Y in range(filtered_pad_ch.shape[0] - k + 1):
            for X in range(filtered_pad_ch.shape[1] - j + 1):
                flt_reg = pad_ch[Y : Y + k, X : X + j] 
                filtered_pad_ch[Y + v_pad, X + h_pad] = np.sum(np.multiply(filter, flt_reg))

        filtered_channel = filtered_pad_ch[
            v_pad : -v_pad if v_pad > 0 else m,
            h_pad : -h_pad if h_pad > 0 else n,
        ]
        filtered_image[:, :, Z] = filtered_channel

    return filtered_image


def create_hybrid_image(
    image1: np.ndarray, image2: np.ndarray, filter: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Takes two images and a low-pass filter and creates a hybrid image. Returns
    the low frequency content of image1, the high frequency content of image 2,
    and the hybrid image.

    Args:
        image1: array of dim (m, n, c)
        image2: array of dim (m, n, c)
        filter: array of dim (x, y)
    Returns:
        low_frequencie: array of shape (m, n, c)
        high_frequencies: array of shape (m, n, c)
        hybrid_image: array of shape (m, n, c)

    HINTS:
    - You will use your my_conv2d_numpy() function in this function.
    - You can get just the high frequency content of an image by removing its
      low frequency content. Think about how to do this in mathematical terms.
    - Don't forget to make sure the pixel values of the hybrid image are
      between 0 and 1. This is known as 'clipping'.
    - If you want to use images with different dimensions, you should resize
      them in the notebook code.
    """

    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    assert image1.shape[2] == image2.shape[2]
    assert filter.shape[0] <= image1.shape[0]
    assert filter.shape[1] <= image1.shape[1]
    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1

    low_frequencies = my_conv2d_numpy(image1, filter)
    high_frequencies = image2 - my_conv2d_numpy(image2, filter)
    hybrid_image = np.clip(low_frequencies + high_frequencies, 0, 1)

    return low_frequencies, high_frequencies, hybrid_image
