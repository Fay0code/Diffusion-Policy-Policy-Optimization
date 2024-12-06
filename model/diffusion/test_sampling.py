import unittest
import tensorflow as tf
import numpy as np
from sampling import cosine_beta_schedule, extract, make_timesteps

class TestSamplingFunctions(unittest.TestCase):

    def test_cosine_beta_schedule(self):
        timesteps = 10
        s = 0.008
        dtype = tf.float32

        # Call the cosine_beta_schedule function
        betas = cosine_beta_schedule(timesteps, s, dtype)

        # Check if the output shape is correct
        self.assertEqual(betas.shape, (timesteps,))

        # Check if the output type is correct
        self.assertEqual(betas.dtype, dtype)

        # Check if the output values are within the expected range
        self.assertTrue(tf.reduce_all(betas >= 0))
        self.assertTrue(tf.reduce_all(betas <= 0.999))

    def test_extract(self):
        # Example tensor `a` containing time step information
        a = tf.constant([0.1, 0.2, 0.3, 0.4, 0.5])

        # Example time step indices `t`
        t = tf.constant([1, 3])

        # Target shape `x_shape`
        x_shape = (2, 3, 3)

        # Call the extract function
        result = extract(a, t, x_shape)

        # Expected output
        expected_output = tf.constant([
            [[0.2], [0.2], [0.2]],
            [[0.4], [0.4], [0.4]]
        ])

        # Check if the output is correct
        tf.debugging.assert_near(result, expected_output)

    def test_make_timesteps(self):
        # Batch size
        batch_size = 5

        # Time step index
        i = 3

        # Device
        device = "/GPU:0"

        # Call the make_timesteps function
        timesteps = make_timesteps(batch_size, i, device)

        # Expected output
        expected_output = tf.constant([3, 3, 3, 3, 3], dtype=tf.int64)

        # Check if the output is correct
        tf.debugging.assert_equal(timesteps, expected_output)

if __name__ == '__main__':
    unittest.main()