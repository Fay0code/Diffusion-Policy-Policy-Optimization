import unittest
import tensorflow as tf
from model.common.critic_tf import CriticObs, CriticObsAct, ViTCritic

class DummyBackbone(tf.keras.layers.Layer):
    def __init__(self, num_patch=16, patch_repr_dim=128):
        super(DummyBackbone, self).__init__()
        self.num_patch = num_patch
        self.patch_repr_dim = patch_repr_dim

    def call(self, inputs, **kwargs):
        return tf.random.normal((inputs.shape[0], self.num_patch, self.patch_repr_dim))

class TestCriticObs(unittest.TestCase):
    def setUp(self):
        self.model = CriticObs(
            cond_dim=11,
            mlp_dims=[256, 256, 256],
            activation_type="Mish",
            use_layernorm=True,
            residual_style=True,
        )

        # Call the model with real tensor data to build it
        # cond = {"state": tf.random.normal((1, 11))}
        cond = {
            "state": tf.random.normal((1, 11)),
            "rgb": tf.random.normal((1, 16, 3, 32, 32))  # Assuming num_patch=16, C=3, H=32, W=32
        }
        self.model(cond)

    def test_initialization(self):
        self.assertIsInstance(self.model, CriticObs)

    def test_call(self):
        cond = tf.random.normal((2, 11))
        output = self.model(cond)
        self.assertEqual(output.shape, (2, 1))

class TestCriticObsAct(unittest.TestCase):
    def setUp(self):
        self.model = CriticObsAct(
            cond_dim=11,
            action_dim=3,
            mlp_dims=[256, 256, 256],
            activation_type="Mish",
            use_layernorm=True,
            residual_style=True,
        )

        # Call the model with real tensor data to build it
        # cond = {"state": tf.random.normal((1, 11))}
        cond = {
            "state": tf.random.normal((1, 11)),
            "rgb": tf.random.normal((1, 16, 3, 32, 32))  # Assuming num_patch=16, C=3, H=32, W=32
        }
        action = tf.random.normal((1, 3))
        self.model(cond, action)

    def test_initialization(self):
        self.assertIsInstance(self.model, CriticObsAct)

    def test_call(self):
        cond = {"state": tf.random.normal((2, 11))}
        action = tf.random.normal((2, 3))
        output = self.model(cond, action)
        if isinstance(output, tuple):
            output = output[0]  # Unpack the tuple if necessary
        self.assertEqual(output.shape, (2,))

class TestViTCritic(unittest.TestCase):
    def setUp(self):
        backbone = DummyBackbone()
        self.model = ViTCritic(
            backbone=backbone,
            cond_dim=11,
            mlp_dims=[256, 256, 256],
            activation_type="Mish",
            use_layernorm=True,
            residual_style=True,
        )

        # Call the model with real tensor data to build it
        cond = {
            "state": tf.random.normal((1, 11)),
            "rgb": tf.random.normal((1, 16, 3, 32, 32))  # Assuming num_patch=16, C=3, H=32, W=32
        }
        self.model(cond)

    def test_initialization(self):
        self.assertIsInstance(self.model, ViTCritic)

    def test_call(self):
        cond = {
            "state": tf.random.normal((2, 11)),
            "rgb": tf.random.normal((2, 16, 3, 32, 32))  # Assuming num_patch=16, C=3, H=32, W=32
        }
        output = self.model(cond)
        self.assertEqual(output.shape, (2,1))

if __name__ == '__main__':
    unittest.main()