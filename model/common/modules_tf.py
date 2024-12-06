"""
Additional implementation of the ViT image encoder from https://github.com/hengyuan-hu/ibrl/tree/main

"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class SpatialEmb(Model):
    def __init__(self, num_patch, patch_dim, prop_dim, proj_dim, dropout):
        super(SpatialEmb, self).__init__()

        proj_in_dim = num_patch + prop_dim
        num_proj = patch_dim
        self.patch_dim = patch_dim
        self.prop_dim = prop_dim

        self.input_proj = tf.keras.Sequential([
            layers.Dense(proj_dim),
            layers.LayerNormalization(),
            layers.ReLU(),
        ])
        self.weight = self.add_weight(shape=(1, num_proj, proj_dim), initializer='random_normal', trainable=True)
        self.dropout = layers.Dropout(dropout)

    def call(self, feat, prop):
        feat = tf.transpose(feat, perm=[0, 2, 1])

        if self.prop_dim > 0:
            repeated_prop = tf.repeat(tf.expand_dims(prop, axis=1), repeats=feat.shape[1], axis=1)
            feat = tf.concat((feat, repeated_prop), axis=-1)

        y = self.input_proj(feat)
        z = tf.reduce_sum(self.weight * y, axis=1)
        z = self.dropout(z)
        return z


class RandomShiftsAug:
    def __init__(self, pad):
        self.pad = pad

    def __call__(self, x):
        n, h, w, c = x.shape
        assert h == w
        padding = [[0, 0], [self.pad, self.pad], [self.pad, self.pad], [0, 0]]
        x = tf.pad(x, padding, mode='REFLECT')
        eps = 1.0 / (h + 2 * self.pad)
        arange = tf.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad)[:h]
        arange = tf.expand_dims(arange, axis=0)
        arange = tf.repeat(arange, repeats=h, axis=0)
        arange = tf.expand_dims(arange, axis=2)
        base_grid = tf.concat([arange, tf.transpose(arange, perm=[1, 0, 2])], axis=2)
        base_grid = tf.expand_dims(base_grid, axis=0)
        base_grid = tf.repeat(base_grid, repeats=n, axis=0)

        shift = tf.random.uniform(shape=(n, 1, 1, 2), minval=0, maxval=2 * self.pad + 1, dtype=tf.int32)
        shift = tf.cast(shift, tf.float32) * 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return tf.keras.layers.experimental.preprocessing.Resizing(h, w)(tf.keras.layers.experimental.preprocessing.Rescaling(1.0)(tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255.0)(tf.keras.layers.experimental.preprocessing.Rescaling(255.0)(tf.keras.layers.experimental.preprocessing.Resizing(h + 2 * self.pad, w + 2 * self.pad)(x)))))

# test random shift
if __name__ == "__main__":
    from PIL import Image
    import requests

    image_url = "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/datacol2_toykitchen7/drawer_pnp/01/2023-04-19_09-18-15/raw/traj_group0/traj0/images0/im_30.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw)
    image = image.resize((96, 96))

    image = np.array(image).astype(np.float32)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    aug = RandomShiftsAug(pad=4)
    image_aug = aug(image)
    image_aug = tf.squeeze(image_aug).numpy().astype(np.uint8)
    image_aug = Image.fromarray(image_aug)
    image_aug.show()
    
    # Save the augmented image
    image_aug.save("augmented_image2.jpg")