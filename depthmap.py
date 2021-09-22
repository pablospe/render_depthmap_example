import math
import numpy as np
from PIL import Image


def write(filename, depth_data):
    # save png image (in millimetres)
    depth_data = (depth_data * 1000).astype(np.uint16)
    image = Image.new("I", depth_data.T.shape)
    image.frombytes(depth_data.tobytes(), 'raw', "I;16")
    image.save(filename)


def read(filename):
    # read png image (in millimetres)
    depth_data = np.asarray(Image.open(filename))
    return depth_data / 1000


def write_compressed(filename, depth_data, max_depth_value=1000):
    """ Write d file in .png form, with clamping the maximum distance to max_depth_value.

    Parameters
    ----------
    filename : string
    depth_data : numpy.ndarray
        depth image data that want to be saved.
    max_depth_value : int, optional
        clamping the maximum distance, by default 1000
    """
    # apply `convert_to_uint16` mapping
    map = lambda x : convert_to_uint16(x, max_depth_value)
    depth_data_compressed = np.vectorize(map)(depth_data).astype(np.uint16)

    # save png image
    image = Image.new("I", depth_data_compressed.T.shape)
    image.frombytes(depth_data_compressed.tobytes(), 'raw', "I;16")
    image.save(filename)


def read_compressed(filename, max_depth_value=1000):
    """ Read a compressed png file representing the quantized depths written by
    write_compressed_depth_map

    Parameters
    ----------
    filename : string

    max_depth_value: int, optional
        max depth value in compressed file. Clamping the maximum distance,
        by default 1000.

    Returns
    -------
    depth_data : numpy.ndarray
        depth image data load from filename.
    """
    # read png image
    depth_data_compressed = np.asarray(Image.open(filename))

    # apply `convert_from_uint16` mapping
    map = lambda x : convert_from_uint16(x, max_depth_value)
    depth_data = np.vectorize(map)(depth_data_compressed)
    return depth_data


def map_from_linear_to_quadratic(linear):
    """
    Apply a quadratic mapping to linear from the .1 to 1.1 x range of the
    quadratic curve (rescaled to an interval of 1) inverted such that lower
    values have greater precision
    """
    a = 1.1
    s = (a - 1) * (a - 1)
    div = 2 * a - 1
    b = a - linear
    non_linear = 1 - (b * b - s) / div
    return non_linear


def map_from_quadratic_to_linear(non_linear):
    """
    Unmap the mapping applied in `map_from_linear_to_quadratic()`
    """
    a = 1.1
    s = (a - 1) * (a - 1)
    div = 2 * a - 1
    linear = a - math.sqrt((1 - non_linear) * div + s)
    return linear


def convert_to_uint16(value, max_depth_value=1000):
    clamped_val = value / max_depth_value

    # Apply a non linear mapping to give closer values more precision
    # This mapping will give values within 100 metres less than half centimeter
    # accuracy, values less than 600 metres less than centimeter accuracy
    # with a maximum error (at 999 metres) of about 4.5 cm accuracy
    non_lin = map_from_linear_to_quadratic(clamped_val) + 7.629510948348211e-06
    non_lin = min([1.0, non_lin])
    quantized_val = non_lin * 65535.0
    return quantized_val


def convert_from_uint16(quantized_value, max_depth_value=1000):
    non_linear = quantized_value / 65535.0
    linear = map_from_quadratic_to_linear(non_linear)
    return linear * max_depth_value
