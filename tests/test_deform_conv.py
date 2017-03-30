import numpy as np
import keras.backend as K
from scipy.ndimage.interpolation import map_coordinates

from deform_conv.deform_conv import (
    tf_map_coordinates,
    sp_batch_map_coordinates, tf_batch_map_coordinates,
    sp_batch_map_offsets, tf_batch_map_offsets
)


def test_tf_map_coordinates():
    np.random.seed(42)
    input = np.random.random((100, 100))
    coords = np.random.random((200, 2)) * 99

    sp_mapped_vals = map_coordinates(input, coords.T, order=1)
    tf_mapped_vals = tf_map_coordinates(
        K.variable(input), K.variable(coords)
    )
    assert np.allclose(sp_mapped_vals, K.eval(tf_mapped_vals), atol=1e-5)


def test_tf_batch_map_coordinates():
    np.random.seed(42)
    input = np.random.random((4, 100, 100))
    coords = np.random.random((4, 200, 2)) * 99

    sp_mapped_vals = sp_batch_map_coordinates(input, coords)
    tf_mapped_vals = tf_batch_map_coordinates(
        K.variable(input), K.variable(coords)
    )
    assert np.allclose(sp_mapped_vals, K.eval(tf_mapped_vals), atol=1e-5)


def test_tf_batch_map_offsets():
    np.random.seed(42)
    input = np.random.random((4, 100, 100))
    offsets = np.random.random((4, 100, 100, 2)) * 2

    sp_mapped_vals = sp_batch_map_offsets(input, offsets)
    tf_mapped_vals = tf_batch_map_offsets(
        K.variable(input), K.variable(offsets)
    )
    assert np.allclose(sp_mapped_vals, K.eval(tf_mapped_vals), atol=1e-5)


def test_tf_batch_map_offsets_grad():
    np.random.seed(42)
    input = np.random.random((4, 100, 100))
    offsets = np.random.random((4, 100, 100, 2)) * 2

    input = K.variable(input)
    offsets = K.variable(offsets)

    tf_mapped_vals = tf_batch_map_offsets(input, offsets)
    grad = K.gradients(tf_mapped_vals, input)[0]
    grad = K.eval(grad)
    assert not np.allclose(grad, 0)
