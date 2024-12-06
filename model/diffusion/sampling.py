import tensorflow as tf
import numpy as np


def cosine_beta_schedule(timesteps, s=0.008, dtype=tf.float32):
    """
    cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return tf.convert_to_tensor(betas_clipped, dtype=dtype)


def extract(a, t, x_shape):
    b = tf.shape(t)[0]
    out = tf.gather(a, t, axis=-1)
    return tf.reshape(out, (b, *((1,) * (len(x_shape) - 1))))


def make_timesteps(batch_size, i,device):
    with tf.device(device):
        t = tf.fill((batch_size,), i)
        t = tf.cast(t, tf.int64)
    return t