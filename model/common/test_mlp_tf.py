import unittest
import tensorflow as tf
from mlp_tf import MLP, ResidualMLP

class TestMLP(unittest.TestCase):

    def setUp(self):
        self.model = MLP(
            dim_list=[10, 20, 30],
            append_dim=5,
            append_layers=[1],
            activation_type="Tanh",
            out_activation_type="Identity",
            use_layernorm=True,
            use_layernorm_final=True,
            dropout=0.1,
            use_drop_final=True,
            verbose=True,
        )

    def test_call(self):
        x = tf.random.normal((1, 10))
        append = tf.random.normal((1, 5))
        output = self.model(x, append=append)
        self.assertEqual(output.shape, (1, 30))

class TestResidualMLP(unittest.TestCase):

    def setUp(self):
        self.model = ResidualMLP(
            dim_list=[10, 20, 20, 20, 30],
            activation_type="Mish",
            out_activation_type="Identity",
            use_layernorm=True,
            use_layernorm_final=True,
            dropout=0,
        )

    def test_call(self):
        x = tf.random.normal((1, 10))
        output = self.model(x)
        self.assertEqual(output.shape, (1, 30))

if __name__ == '__main__':
    unittest.main()