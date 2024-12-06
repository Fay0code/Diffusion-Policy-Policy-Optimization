import unittest
import tensorflow as tf
import numpy as np
from modules_tf import SpatialEmb, RandomShiftsAug

class TestSpatialEmb(unittest.TestCase):
    def setUp(self):
        self.num_patch = 16
        self.patch_dim = 64
        self.prop_dim = 32
        self.proj_dim = 128
        self.dropout = 0.1
        self.model = SpatialEmb(self.num_patch, self.patch_dim, self.prop_dim, self.proj_dim, self.dropout)

    def test_call(self):
        feat = tf.random.normal((2, self.num_patch, self.patch_dim))
        prop = tf.random.normal((2, self.prop_dim))
        output = self.model(feat, prop)
        self.assertEqual(output.shape, (2, self.proj_dim))  # 修正预期输出形状

class TestRandomShiftsAug(unittest.TestCase):
    def setUp(self):
        self.pad = 4
        self.augmentor = RandomShiftsAug(self.pad)

    def test_call(self):
        image = tf.random.uniform((2, 32, 32, 3), minval=0, maxval=255, dtype=tf.float32)
        augmented_image = self.augmentor(image)
        self.assertEqual(augmented_image.shape, (2, 32, 32, 3))

if __name__ == '__main__':
    unittest.main()