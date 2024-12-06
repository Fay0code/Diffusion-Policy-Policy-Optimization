"""
Implementation of Multi-layer Perception (MLP).

Residual model is taken from https://github.com/ALRhub/d3il/blob/main/agents/models/common/mlp.py

"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from collections import OrderedDict
import logging
import tensorflow_addons as tfa

activation_dict = {
    "ReLU": layers.ReLU(),
    "ELU": layers.ELU(),
    "GELU": tfa.activations.gelu,
    "Tanh": layers.Activation('tanh'),
    "Mish": layers.Activation(lambda x: x * tf.math.tanh(tf.math.softplus(x))),
    "Identity": layers.Activation('linear'),
    "Softplus": layers.Activation(tf.nn.softplus),
}

class MLP(Model):
    def __init__(
        self,
        dim_list,
        append_dim=0,
        append_layers=None,
        activation_type="Tanh",
        out_activation_type="Identity",
        use_layernorm=False,
        use_layernorm_final=False,
        dropout=0,
        use_drop_final=False,
        verbose=False,
    ):
        super(MLP, self).__init__()

        self.layers_list = []
        self.append_layers = append_layers
        num_layer = len(dim_list) - 1
        for idx in range(num_layer):
            i_dim = dim_list[idx]
            o_dim = dim_list[idx + 1]
            if append_dim > 0 and idx in append_layers:
                i_dim += append_dim
            linear_layer = layers.Dense(o_dim)

            # Add module components
            layer_components = [linear_layer]
            if use_layernorm and (idx < num_layer - 1 or use_layernorm_final):
                layer_components.append(layers.LayerNormalization())
            if dropout > 0 and (idx < num_layer - 1 or use_drop_final):
                layer_components.append(layers.Dropout(dropout))

            # add activation function
            act = (
                activation_dict[activation_type]
                if idx != num_layer - 1
                else activation_dict[out_activation_type]
            )
            layer_components.append(act)

            # re-construct module
            self.layers_list.append(layer_components)
        if verbose:
            logging.info(self.layers_list)
    @tf.function
    def call(self, x, append=None):
        for layer_ind, layer_components in enumerate(self.layers_list):
            if append is not None and layer_ind in self.append_layers:
                x = tf.concat([x, append], axis=-1)
            for component in layer_components:
                x = component(x)
        return x

class ResidualMLP(Model):
    """
    Simple multi layer perceptron network with residual connections for
    benchmarking the performance of different networks. The resiudal layers
    are based on the IBC paper implementation, which uses 2 residual lalyers
    with pre-actication with or without dropout and normalization.
    """

    def __init__(
        self,
        dim_list,
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        use_layernorm_final=False,
        dropout=0,
    ):
        super(ResidualMLP, self).__init__()
        hidden_dim = dim_list[1]
        num_hidden_layers = len(dim_list) - 3
        assert num_hidden_layers % 2 == 0
        self.layers_list = [layers.Dense(hidden_dim)]
        self.layers_list.extend(
            [
                TwoLayerPreActivationResNetLinear(
                    hidden_dim=hidden_dim,
                    activation_type=activation_type,
                    use_layernorm=use_layernorm,
                    dropout=dropout,
                )
                for _ in range(1, num_hidden_layers, 2)
            ]
        )
        self.layers_list.append(layers.Dense(dim_list[-1]))
        if use_layernorm_final:
            self.layers_list.append(layers.LayerNormalization())
        self.layers_list.append(activation_dict[out_activation_type])

    @tf.function
    def call(self, x):
        for layer in self.layers_list:
            x = layer(x)
        return x

class TwoLayerPreActivationResNetLinear(layers.Layer):
    def __init__(
        self,
        hidden_dim,
        activation_type="Mish",
        use_layernorm=False,
        dropout=0,
    ):
        super(TwoLayerPreActivationResNetLinear, self).__init__()
        self.l1 = layers.Dense(hidden_dim)
        self.l2 = layers.Dense(hidden_dim)
        self.act = activation_dict[activation_type]
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.norm1 = layers.LayerNormalization(epsilon=1e-06)
            self.norm2 = layers.LayerNormalization(epsilon=1e-06)
        if dropout > 0:
            raise NotImplementedError("Dropout not implemented for residual MLP!")

    @tf.function
    def call(self, x):
        x_input = x
        if self.use_layernorm:
            x = self.norm1(x)
        x = self.l1(self.act(x))
        if self.use_layernorm:
            x = self.norm2(x)
        x = self.l2(self.act(x))
        return x + x_input