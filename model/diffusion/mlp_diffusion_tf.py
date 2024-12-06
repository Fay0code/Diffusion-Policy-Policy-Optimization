"""
MLP models for diffusion policies.

"""
import copy
import tensorflow as tf
from tensorflow.keras import layers, Model
import logging
import einops
from copy import deepcopy
from model.common.mlp_tf import MLP, ResidualMLP
from model.diffusion.modules_tf import SinusoidalPosEmb
from model.common.modules_tf import SpatialEmb, RandomShiftsAug

log = logging.getLogger(__name__)

class VisionDiffusionMLP(Model):
    """With ViT backbone"""

    def __init__(
        self,
        backbone,
        action_dim,
        horizon_steps,
        cond_dim,
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
    ):
        super(VisionDiffusionMLP, self).__init__()

        # vision
        self.backbone = backbone
        if augment:
            self.aug = RandomShiftsAug(pad=4)
        self.augment = augment
        self.num_img = num_img
        self.img_cond_steps = img_cond_steps
        if spatial_emb > 0:
            assert spatial_emb > 1, "this is the dimension"
            if num_img > 1:
                self.compress1 = SpatialEmb(
                    num_patch=self.backbone.num_patch,
                    patch_dim=self.backbone.patch_repr_dim,
                    prop_dim=cond_dim,
                    proj_dim=spatial_emb,
                    dropout=dropout,
                )
                self.compress2 = deepcopy(self.compress1)
            else:  # TODO: clean up
                self.compress = SpatialEmb(
                    num_patch=self.backbone.num_patch,
                    patch_dim=self.backbone.patch_repr_dim,
                    prop_dim=cond_dim,
                    proj_dim=spatial_emb,
                    dropout=dropout,
                )
            visual_feature_dim = spatial_emb * num_img
        else:
            self.compress = tf.keras.Sequential([
                layers.Dense(visual_feature_dim),
                layers.LayerNormalization(),
                layers.Dropout(dropout),
                layers.ReLU(),
            ])

        # diffusion
        input_dim = (
            time_dim + action_dim * horizon_steps + visual_feature_dim + cond_dim
        )
        output_dim = action_dim * horizon_steps
        self.time_embedding = tf.keras.Sequential([
            SinusoidalPosEmb(time_dim),
            layers.Dense(time_dim * 2),
            layers.Activation('mish'),
            layers.Dense(time_dim),
        ])
        if residual_style:
            model = ResidualMLP
        else:
            model = MLP
        self.mlp_mean = model(
            [input_dim] + mlp_dims + [output_dim],
            activation_type=activation_type,
            out_activation_type=out_activation_type,
            use_layernorm=use_layernorm,
        )
        self.time_dim = time_dim
        
    @tf.function
    def call(
        self,
        x,
        time,
        cond: dict,
        **kwargs,
    ):
        """
        x: (B, Ta, Da)
        time: (B,) or int, diffusion step
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
            rgb: (B, To, C, H, W)

        TODO long term: more flexible handling of cond
        """
        B, Ta, Da = x.shape
        _, T_rgb, C, H, W = cond["rgb"].shape

        # flatten chunk
        x = tf.reshape(x, (B, -1))

        # flatten history
        state = tf.reshape(cond["state"], (B, -1))

        # Take recent images --- sometimes we want to use fewer img_cond_steps than cond_steps (e.g., 1 image but 3 prio)
        rgb = cond["rgb"][:, -self.img_cond_steps:]

        # concatenate images in cond by channels
        if self.num_img > 1:
            rgb = tf.reshape(rgb, (B, T_rgb, self.num_img, 3, H, W))
            rgb = einops.rearrange(rgb, "b t n c h w -> b n (t c) h w")
        else:
            rgb = einops.rearrange(rgb, "b t c h w -> b (t c) h w")

        # convert rgb to float32 for augmentation
        rgb = tf.cast(rgb, tf.float32)

        # get vit output - pass in two images separately
        if self.num_img > 1:  # TODO: properly handle multiple images
            rgb1 = rgb[:, 0]
            rgb2 = rgb[:, 1]
            if self.augment:
                rgb1 = self.aug(rgb1)
                rgb2 = self.aug(rgb2)
            feat1 = self.backbone(rgb1)
            feat2 = self.backbone(rgb2)
            feat1 = self.compress1(feat1, state)
            feat2 = self.compress2(feat2, state)
            feat = tf.concat([feat1, feat2], axis=-1)
        else:  # single image
            if self.augment:
                rgb = self.aug(rgb)
            feat = self.backbone(rgb)

            # compress
            if isinstance(self.compress, SpatialEmb):
                feat = self.compress(feat, state)
            else:
                feat = tf.reshape(feat, (B, -1))
                feat = self.compress(feat)
        cond_encoded = tf.concat([feat, state], axis=-1)

        # append time and cond
        time = tf.reshape(time, (B, 1))
        time_emb = tf.reshape(self.time_embedding(time), (B, self.time_dim))
        x = tf.concat([x, time_emb, cond_encoded], axis=-1)

        # mlp
        out = self.mlp_mean(x)
        return tf.reshape(out, (B, Ta, Da))


class DiffusionMLP(Model):
    #input some parameters, and build a model, but no real input value
    def __init__(
        self,
        action_dim,
        horizon_steps,
        cond_dim,
        time_dim=16,
        mlp_dims=[256, 256],
        cond_mlp_dims=None,
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        residual_style=False,
    ):
        super(DiffusionMLP, self).__init__()
        output_dim = action_dim * horizon_steps
        
        #encode time information into high dim space
        #out dim = time_dim
        self.time_embedding = tf.keras.Sequential([
            SinusoidalPosEmb(time_dim),
            layers.Dense(time_dim * 2),
            layers.Activation('mish'),
            layers.Dense(time_dim),
        ])

        #choose which mlp to use
        if residual_style:
            model = ResidualMLP
        else:
            model = MLP
        if cond_mlp_dims is not None:
            self.cond_mlp = MLP(
                [cond_dim] + cond_mlp_dims,
                activation_type=activation_type,
                out_activation_type="Identity",
            )
            input_dim = time_dim + action_dim * horizon_steps + cond_mlp_dims[-1]
        else:
            input_dim = time_dim + action_dim * horizon_steps + cond_dim

        #call the mlp
        # two layer mlp with input_dim, mlp_dims, output_dim
        #output dim = action_dim * horizon_steps            
        self.mlp_mean = model(
            [input_dim] + mlp_dims + [output_dim],
            activation_type=activation_type,
            out_activation_type=out_activation_type,
            use_layernorm=use_layernorm,
        )
        self.time_dim = time_dim

    @tf.function
    def call(
        self,
        x,
        time,
        cond,
        **kwargs,
    ):
        """
        x: (B, Ta, Da)
        time: (B,) or int, diffusion step
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
        """

        B, Ta, Da = x.shape

        if B == None:
            return x

        
        # flatten chunk
        x = tf.reshape(x, (B, -1))

        # flatten history
        state = tf.reshape(cond["state"], (B, -1))

        # obs encoder
        if hasattr(self, "cond_mlp"):
            state = self.cond_mlp(state)

        # append time and cond
        time = tf.reshape(time, (B, 1))
        time_emb = tf.reshape(self.time_embedding(time), (B, self.time_dim))
        x = tf.concat([x, time_emb, state], axis=-1)

        # mlp head
        out = self.mlp_mean(x)
        return tf.reshape(out, (B, Ta, Da))

# # 创建模型实例
# model = DiffusionMLP(
#     action_dim=3,
#     horizon_steps=4,
#     cond_dim=11,
#     time_dim=16,
#     mlp_dims=[256, 256, 256],
#     activation_type="Mish",
#     out_activation_type="Identity",
#     use_layernorm=False,
#     residual_style=True

# # 创建输入张量
# x = tf.random.normal((1, 4, 3))
# time = tf.constant([1], dtype=tf.float32)
# cond = {
#     "state": tf.random.normal((1, 4, 11)),
#     "rgb": tf.random.normal((1, 4, 3, 32, 32))
# }

# # 调用模型，自动构建网络
# outputs = model(x, time, cond)

# # 克隆模型并复制权重
# actor_ft = tf.keras.models.clone_model(model)
# actor_ft(x, time, cond) 
# # actor_ft.set_weights(model.get_weights())

# # 计算模型参数数量
# num_params = model.count_params()
# num_params_ft = actor_ft.count_params()
# print(f"Number of parameters in the model: {num_params}")
# print(f"Number of parameters in the cloned model: {num_params_ft}")