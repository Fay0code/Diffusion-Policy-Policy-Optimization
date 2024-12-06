import unittest
import tensorflow as tf
from mlp_diffusion_tf import VisionDiffusionMLP, DiffusionMLP
from tensorflow.keras import layers

class DummyBackbone(tf.keras.Model):
    def __init__(self):
        super(DummyBackbone, self).__init__()
        self.dense = layers.Dense(128)

    def call(self, x):
        return self.dense(x)

class TestVisionDiffusionMLP(unittest.TestCase):
    def setUp(self):
        self.backbone = DummyBackbone()
        self.model = VisionDiffusionMLP(
            backbone=self.backbone,
            action_dim=3,
            horizon_steps=4,
            cond_dim=11,
            img_cond_steps=1,
            time_dim=16,
            mlp_dims=[256, 256],
            activation_type="Mish",
            out_activation_type="Identity",
            use_layernorm=False,
            residual_style=False,
            spatial_emb=0,
            visual_feature_dim=128,
            dropout=0,
            num_img=1,
            augment=False,
        )

    def test_call(self):
        x = tf.random.normal((2, 4, 3))
        time = tf.constant([1, 10], dtype=tf.float32)
        cond = {
            "state": tf.random.normal((2, 4, 11)),
            "rgb": tf.random.normal((2, 4, 3, 32, 32))
        }
        output = self.model(x, time, cond)
        self.assertEqual(output.shape, (2, 4, 3))

class TestDiffusionMLP(unittest.TestCase):
    def setUp(self):
        self.model = DiffusionMLP(
            action_dim=3,
            horizon_steps=4,
            cond_dim=11,
            time_dim=16,
            mlp_dims=[256, 256],
            cond_mlp_dims=None,
            activation_type="Mish",
            out_activation_type="Identity",
            use_layernorm=False,
            residual_style=False,
        )

    def test_call(self):
        x = tf.random.normal((2, 4, 3))
        time = tf.constant([1, 10], dtype=tf.float32)
        cond = {
            "state": tf.random.normal((2, 4, 11)),
        }
        output = self.model(x, time, cond)
        self.assertEqual(output.shape, (2, 4, 3))

if __name__ == '__main__':
    unittest.main()