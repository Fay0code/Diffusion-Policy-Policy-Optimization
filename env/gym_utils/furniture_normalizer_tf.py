"""
Normalization for Furniture-Bench environments.

TODO: use this normalizer for all benchmarks.

"""

import tensorflow as tf


class LinearNormalizer(tf.Module):
    def __init__(self):
        super().__init__()
        self.stats = {}

    def fit(self, data_dict):
        for key, tensor in data_dict.items():
            min_value = tf.reduce_min(tensor, axis=0)
            max_value = tf.reduce_max(tensor, axis=0)

            # Check if any column has only one value throughout
            diff = max_value - min_value
            constant_columns = tf.equal(diff, 0)

            # Set a small range for constant columns to avoid division by zero
            min_value = tf.where(constant_columns, min_value - 1, min_value)
            max_value = tf.where(constant_columns, max_value + 1, max_value)

            self.stats[key] = {
                "min": tf.Variable(min_value, trainable=False),
                "max": tf.Variable(max_value, trainable=False),
            }

    def _normalize(self, x, key):
        stats = self.stats[key]
        x = (x - stats["min"]) / (stats["max"] - stats["min"])
        x = 2 * x - 1
        return x

    def _denormalize(self, x, key):
        stats = self.stats[key]
        x = (x + 1) / 2
        x = x * (stats["max"] - stats["min"]) + stats["min"]
        return x

    def __call__(self, x, key, forward=True):
        if forward:
            return self._normalize(x, key)
        else:
            return self._denormalize(x, key)

    def load_state_dict(self, state_dict):
        stats = {}
        for key, value in state_dict.items():
            if key.startswith("stats."):
                param_key = key[6:]
                keys = param_key.split(".")
                current_dict = stats
                for k in keys[:-1]:
                    if k not in current_dict:
                        current_dict[k] = {}
                    current_dict = current_dict[k]
                current_dict[keys[-1]] = tf.Variable(value, trainable=False)

        self.stats = stats

        return f"<Added keys {self.stats.keys()} to the normalizer.>"

    def keys(self):
        return self.stats.keys()