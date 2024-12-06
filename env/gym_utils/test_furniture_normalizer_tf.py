import unittest
import numpy as np
import tensorflow as tf
from furniture_normalizer_tf import LinearNormalizer

class TestLinearNormalizer(unittest.TestCase):
    def setUp(self):
        self.normalizer = LinearNormalizer()
        self.data_dict = {
            "state": tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=tf.float32),
            "action": tf.constant([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=tf.float32),
        }

    def test_fit(self):
        self.normalizer.fit(self.data_dict)
        
        self.assertIn("state", self.normalizer.stats)
        self.assertIn("action", self.normalizer.stats)
        self.assertTrue(tf.reduce_all(self.normalizer.stats["state"]["min"] == tf.constant([1.0, 2.0])))
        self.assertTrue(tf.reduce_all(self.normalizer.stats["state"]["max"] == tf.constant([5.0, 6.0])))
        self.assertTrue(tf.reduce_all(self.normalizer.stats["action"]["min"] == tf.constant([0.1, 0.2])))
        self.assertTrue(tf.reduce_all(self.normalizer.stats["action"]["max"] == tf.constant([0.5, 0.6])))

    def test_normalize(self):
        self.normalizer.fit(self.data_dict)
        
        x = tf.constant([[3.0, 4.0]], dtype=tf.float32)
        normalized_x = self.normalizer(x, "state", forward=True)
        expected_normalized_x = tf.constant([[0.0, 0.0]], dtype=tf.float32)
        
        self.assertTrue(tf.reduce_all(tf.abs(normalized_x - expected_normalized_x) < 1e-6))

    def test_denormalize(self):
        self.normalizer.fit(self.data_dict)
        
        x = tf.constant([[0.0, 0.0]], dtype=tf.float32)
        denormalized_x = self.normalizer(x, "state", forward=False)
        expected_denormalized_x = tf.constant([[3.0, 4.0]], dtype=tf.float32)
        
        self.assertTrue(tf.reduce_all(tf.abs(denormalized_x - expected_denormalized_x) < 1e-6))

    def test_load_state_dict(self):
        state_dict = {
            "stats.state.min": tf.constant([1.0, 2.0], dtype=tf.float32),
            "stats.state.max": tf.constant([5.0, 6.0], dtype=tf.float32),
        }
        self.normalizer.load_state_dict(state_dict)
        
        self.assertIn("state", self.normalizer.stats)
        self.assertTrue(tf.reduce_all(self.normalizer.stats["state"]["min"] == tf.constant([1.0, 2.0])))
        self.assertTrue(tf.reduce_all(self.normalizer.stats["state"]["max"] == tf.constant([5.0, 6.0])))

    def test_keys(self):
        self.normalizer.fit(self.data_dict)
        
        keys = self.normalizer.keys()
        self.assertIn("state", keys)
        self.assertIn("action", keys)

if __name__ == "__main__":
    unittest.main()