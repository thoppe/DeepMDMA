import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim


def cppn(
        width,
        height,
        batch=1,
        num_output_channels=3,
        num_hidden_channels=24,
        num_layers=8,
        activation_func=_composite_activation,
        normalize=False,
):
    """Compositional Pattern Producing Network
    Args:
      width: width of resulting image, equals height
      batch: batch dimension of output, note that all params share the same weights!
      num_output_channels:
      num_hidden_channels:
      num_layers:
      activation_func:
      normalize:
    Returns:
      The collapsed shape, represented as a list.
    """
    r = 3.0 ** 0.5  # std(coord_range) == 1.0
    coord_range_width = tf.linspace(-r, r, width)
    coord_range_height = tf.linspace(-r, r, width)
    
    y, x = tf.meshgrid(coord_range_width, coord_range_height, indexing="ij")
    net = tf.stack([tf.stack([x, y], -1)] * batch, 0)

    with slim.arg_scope(
        [slim.conv2d],
        kernel_size=[1, 1],
        activation_fn=None,
        weights_initializer=tf.initializers.variance_scaling(),
        biases_initializer=tf.initializers.random_normal(0.0, 0.1),
    ):
        for i in range(num_layers):
            x = slim.conv2d(net, num_hidden_channels)
            if normalize:
                x = slim.instance_norm(x)
            net = activation_func(x)
        rgb = slim.conv2d(
            net,
            num_output_channels,
            activation_fn=tf.nn.sigmoid,
            weights_initializer=tf.zeros_initializer(),
        )
    return rgb
