import tensorflow as tf


def fully_connected(x, n_output, name="fully_connected"):
        with tf.variable_scope("FullyConnected" + str(name), reuse=tf.AUTO_REUSE):
            if len(x.get_shape()) != 2:
                x = flatten(x, reuse=tf.AUTO_REUSE)

            W = tf.get_variable(
                name='W',
                shape=[x.get_shape().as_list()[1], n_output],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer()
            )

            b = tf.get_variable(
                name='b',
                shape=[n_output],
                dtype=tf.float32,
                initializer=tf.constant_initializer(0.0)
            )

            h = tf.nn.bias_add(
                name='h',
                value=tf.matmul(x, W),
                bias=b
            )

            return h, W


def flatten(x, name=None, reuse=None):
    """Flatten Tensor to 2-dimensions.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor to flatten.
    name : None, optional
        Variable scope for flatten operations

    Returns
    -------
    flattened : tf.Tensor
        Flattened tensor.
    """
    with tf.variable_scope('flatten'):
        dims = x.get_shape().as_list()
        if len(dims) == 4:
            flattened = tf.reshape(
                x,
                shape=[-1, dims[1] * dims[2] * dims[3]])
        elif len(dims) == 2 or len(dims) == 1:
            flattened = x
        else:
            raise ValueError('Expected n dimensions of 1, 2 or 4.  Found:',
                             len(dims))

        return flattened


def binary_cross_entropy(z, x):
    eps = 1e-12
    return -(x * tf.log(z + eps) + (1. - x) * tf.log(1. - z + eps))
