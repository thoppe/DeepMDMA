from lucid.optvis.param.cppn import _composite_activation
import tensorflow as tf
from tensorflow.contrib import slim

def reduced_cppn(width):

    r = 3.0 ** 0.5  # std(coord_range) == 1.0
    coord_range = tf.linspace(-r, r, width)
    y, x = tf.meshgrid(coord_range, coord_range, indexing="ij")
    net = tf.stack([tf.stack([x, y], -1)], 0)

    return net


def add_layer(
    net_in,
    num_output_channels=3,
    num_hidden_channels=24,
    activation_fn=_composite_activation,
):

    net = slim.conv2d(
        net_in,
        num_hidden_channels,
        kernel_size=[1, 1],
        activation_fn=activation_fn,
        weights_initializer=tf.initializers.variance_scaling(),
        biases_initializer=tf.initializers.random_normal(0.0, 0.1),
    )

    return net




def create_locked_network(
        batch_size,
        network_size,
        num_layers=8,
        num_shared_layers=4,
        num_hidden_channels=24,
        activation_fn=_composite_activation,
):
    nets = []

    with tf.variable_scope(f"CPPN_shared"):
        shared_net = reduced_cppn(network_size)

        for i in range(num_shared_layers):
            shared_net = add_layer(shared_net)

    for k in range(batch_size):
        with tf.variable_scope(f"CPPN_layer_{k}"):

            rgb = shared_net
            
            for i in range(num_layers-num_shared_layers):
                rgb = add_layer(rgb)

            # Final Layer
            rgb = slim.conv2d(
                rgb, 3, kernel_size=[1,1],
                activation_fn=tf.nn.sigmoid,
                weights_initializer=tf.zeros_initializer(),
                biases_initializer=tf.initializers.random_normal(0.0, 0.1),
            )
            
            nets.append(rgb)

    net = tf.concat(nets, axis=0)
    return net
