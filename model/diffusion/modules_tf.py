"""
From Diffuser https://github.com/jannerm/diffuser

For MLP and UNet diffusion models.

"""

import math
import tensorflow as tf
from tensorflow.keras import layers, Model

class SinusoidalPosEmb(layers.Layer):
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def call(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = tf.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
        emb = tf.expand_dims(x, -1) * tf.expand_dims(emb, 0)
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        return emb

class Downsample1d(layers.Layer):
    def __init__(self, dim):
        super(Downsample1d, self).__init__()
        self.conv = layers.Conv1D(dim, 3, strides=2, padding='same')

    def call(self, x):
        return self.conv(x)

class Upsample1d(layers.Layer):
    def __init__(self, dim):
        super(Upsample1d, self).__init__()
        self.conv = layers.Conv1DTranspose(dim, 4, strides=2, padding='same')

    def call(self, x):
        return self.conv(x)

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

class Conv1dBlock(layers.Layer):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(
        self,
        inp_channels,
        out_channels,
        kernel_size,
        n_groups=None,
        activation_type="Mish",
        eps=1e-5,
    ):
        super(Conv1dBlock, self).__init__()
        if activation_type == "Mish":
            act = tf.keras.layers.Activation(mish)
        elif activation_type == "ReLU":
            act = tf.keras.layers.Activation(tf.nn.relu)
        else:
            raise ValueError("Unknown activation type for Conv1dBlock")

        self.conv = layers.Conv1D(out_channels, kernel_size, padding='same')
        self.rearrange1 = layers.Reshape((-1, 1, out_channels)) if n_groups is not None else tf.identity
        self.group_norm = layers.LayerNormalization(axis=-1, epsilon=eps) if n_groups is not None else tf.identity
        self.rearrange2 = layers.Reshape((-1, out_channels)) if n_groups is not None else tf.identity
        self.act = act

    def call(self, x):
        x = self.conv(x)
        if callable(self.rearrange1):
            x = self.rearrange1(x)
        if callable(self.group_norm):
            x = self.group_norm(x)
        if callable(self.rearrange2):
            x = self.rearrange2(x)
        x = self.act(x)
        return x