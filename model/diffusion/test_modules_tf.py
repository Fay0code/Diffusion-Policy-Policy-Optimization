import unittest
import tensorflow as tf
from modules_tf import SinusoidalPosEmb, Downsample1d, Upsample1d, Conv1dBlock

class TestSinusoidalPosEmb(unittest.TestCase):

    def setUp(self):
        self.pos_emb = SinusoidalPosEmb(dim=16)

    def test_call(self):
        x = tf.constant([1.0, 2.0, 3.0])
        output = self.pos_emb(x)
        self.assertEqual(output.shape, (3, 16))

class TestDownsample1d(unittest.TestCase):

    def setUp(self):
        self.downsample = Downsample1d(dim=16)

    def test_call(self):
        x = tf.random.normal((1, 64, 16))
        output = self.downsample(x)
        self.assertEqual(output.shape, (1, 32, 16))

class TestUpsample1d(unittest.TestCase):

    def setUp(self):
        self.upsample = Upsample1d(dim=16)

    def test_call(self):
        x = tf.random.normal((1, 32, 16))
        output = self.upsample(x)
        self.assertEqual(output.shape, (1, 64, 16))

class TestConv1dBlock(unittest.TestCase):

    def setUp(self):
        self.conv_block = Conv1dBlock(inp_channels=16, out_channels=32, kernel_size=3, n_groups=4)

    def test_call(self):
        x = tf.random.normal((1, 64, 16))
        output = self.conv_block(x)
        self.assertEqual(output.shape, (1, 64, 32))

if __name__ == '__main__':
    unittest.main()