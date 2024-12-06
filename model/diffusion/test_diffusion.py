import unittest
import tensorflow as tf
from tensorflow.keras import layers
from diffusion import DiffusionModel, Sample
from model.common.mlp_tf import MLP
from mlp_diffusion_tf import DiffusionMLP

class TestDiffusionModel(unittest.TestCase):
    def setUp(self):
        self.network = DiffusionMLP(
            action_dim=3,
            horizon_steps=4,
            cond_dim=11,
            time_dim=16,
            mlp_dims=[512, 512, 512],
            cond_mlp_dims=None,
            activation_type="ReLU",
            out_activation_type="Identity",
            use_layernorm=False,
            residual_style=True
        )
        
        # Call the model with real tensor data to build it
        x = tf.random.normal((2, 4, 3))
        time = tf.constant([1,10], dtype=tf.float32)
        cond = {
            "state": tf.random.normal((2, 4, 11)),
            "rgb": tf.random.normal((2, 4, 3, 32, 32))
        }
        self.network(x, time, cond)
        
        self.model = DiffusionModel(
            network=self.network,
            horizon_steps=4,
            obs_dim=11,
            action_dim=3,
            denoising_steps=20,
            predict_epsilon=True,
            use_ddim=False
        )

    def test_p_mean_var(self):
        x = tf.random.normal((2, 4, 3))
        t = tf.constant([1, 5 ], dtype=tf.int32)
        cond = {'state': tf.random.normal((2, 4, 11))}
        index = tf.constant([0, 1], dtype=tf.int32)
        mu, logvar = self.model.p_mean_var(x, t, cond, index)
        self.assertEqual(mu.shape, x.shape)
        self.assertEqual(logvar.shape, (2,1,1))

    def test_call(self):
        cond = {'state': tf.random.normal((2, 4, 11))}
        sample = self.model(cond)
        self.assertEqual(sample.trajectories.shape, (2, 4, 3))

    def test_loss(self):
        x = tf.random.normal((2, 4, 3))
        cond = {'state': tf.random.normal((2, 4, 11))}
        loss = self.model.loss(x, cond)
        self.assertIsInstance(loss, tf.Tensor)

    def test_p_losses(self):
        x_start = tf.random.normal((2, 4, 3))
        cond = {'state': tf.random.normal((2, 4, 11))}
        t = tf.constant([1, 5], dtype=tf.int32)
        loss = self.model.p_losses(x_start, cond, t)
        self.assertIsInstance(loss, tf.Tensor)

    def test_q_sample(self):
        x_start = tf.random.normal((2, 4, 3))
        t = tf.constant([1, 5], dtype=tf.int32)
        x_noisy = self.model.q_sample(x_start, t)
        self.assertEqual(x_noisy.shape, x_start.shape)

if __name__ == '__main__':
    unittest.main()