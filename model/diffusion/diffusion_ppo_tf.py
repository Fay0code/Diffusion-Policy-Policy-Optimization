"""
DPPO: Diffusion Policy Policy Optimization. 

K: number of denoising steps
To: observation sequence length
Ta: action chunk size
Do: observation dimension
Da: action dimension

C: image channels
H, W: image height and width

"""
from typing import Optional
import tensorflow as tf
import tensorflow_probability as tfp
import logging
import math
from diffusion_vpg_tf import VPGDiffusion

log = logging.getLogger(__name__)

class PPODiffusion(VPGDiffusion):
    """
    实现基于扩散的近端策略优化（PPO）算法变体
    继承自VPGDiffusion,该类提供了底层的diffusion策略梯度实现
    """

    def __init__(
        self,
        gamma_denoising: float,
        clip_ploss_coef: float,
        clip_ploss_coef_base: float = 1e-3,
        clip_ploss_coef_rate: float = 3,
        clip_vloss_coef: Optional[float] = None,
        clip_advantage_lower_quantile: float = 0,
        clip_advantage_upper_quantile: float = 1,
        norm_adv: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Whether to normalize advantages within batch
        self.norm_adv = norm_adv

        # Clipping value for policy loss
        self.clip_ploss_coef = clip_ploss_coef
        self.clip_ploss_coef_base = clip_ploss_coef_base
        self.clip_ploss_coef_rate = clip_ploss_coef_rate

        # Clipping value for value loss
        self.clip_vloss_coef = clip_vloss_coef

        # Discount factor for diffusion MDP
        self.gamma_denoising = gamma_denoising

        # Quantiles for clipping advantages
        self.clip_advantage_lower_quantile = clip_advantage_lower_quantile 
        self.clip_advantage_upper_quantile = clip_advantage_upper_quantile

    def loss(
        self,
        obs,
        chains_prev,
        chains_next,
        denoising_inds,
        returns,
        oldvalues,
        advantages,
        oldlogprobs,
        use_bc_loss=False,
        reward_horizon=4,
    ):
        """
        相对于VPGDiffusion,添加了PPO loss的计算
        PPO loss 损失计算

        参数：
        obs: 包含状态和图像的字典，最后的观察在最后，dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
            rgb: (B, To, C, H, W)
        chains: (B, K+1, Ta, Da)
        returns: (B, )
        values: (B, )
        advantages: (B,)
        oldlogprobs: (B, K, Ta, Da)
        use_bc_loss: whether to add BC regularization loss 是否添加行为克隆正则化损失
        reward_horizon: action horizon that backpropagates gradient 反向传播梯度的动作时域长度
        """
        # Get new logprobs for denoising steps from T-1 to 0 - entropy is fixed for diffusion
        newlogprobs, eta = self.get_logprobs_subsample(
            obs,
            chains_prev,
            chains_next,
            denoising_inds,
            get_ent=True,
        )
        


        entropy_loss = -tf.reduce_mean(eta)
        newlogprobs = tf.clip_by_value(newlogprobs, clip_value_min=-5, clip_value_max=2)
        oldlogprobs = tf.clip_by_value(oldlogprobs, clip_value_min=-5, clip_value_max=2)
    
        # only backpropagate through the earlier steps (e.g., ones actually executed in the environment)
        newlogprobs = newlogprobs[:, :reward_horizon, :]
        oldlogprobs = oldlogprobs[:, :reward_horizon, :]

        
        # Get the logprobs - batch over B and denoising steps
        newlogprobs = tf.reduce_mean(newlogprobs, axis=(-1, -2))
        oldlogprobs = tf.reduce_mean(oldlogprobs, axis=(-1, -2))
        # return newlogprobs, oldlogprobs
    
        bc_loss = 0
        if use_bc_loss:
            samples = self.call(
                cond=obs,
                deterministic=False,
                return_chain=True,
                use_base_policy=True,
            )
            bc_logprobs = self.get_logprobs(
                obs,
                samples.chains,
                get_ent=False,
                use_base_policy=False,
            )
            bc_logprobs = tf.clip_by_value(bc_logprobs, clip_value_min=-5, clip_value_max=2)
            bc_logprobs = tf.reduce_mean(bc_logprobs, axis=(-1, -2))
            bc_loss = -tf.reduce_mean(bc_logprobs)

        if self.norm_adv:
            advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)

        advantage_min = tfp.stats.percentile(advantages, self.clip_advantage_lower_quantile * 100.0)
        advantage_max = tfp.stats.percentile(advantages, self.clip_advantage_upper_quantile * 100.0)
        advantages = tf.clip_by_value(advantages, clip_value_min=advantage_min, clip_value_max=advantage_max)

        
        # discount = tf.map_fn(
        #     lambda i: self.gamma_denoising ** (self.ft_denoising_steps -tf.cast(i, tf.float32) - 1),
        #     denoising_inds,
        #     dtype=tf.float32
        # )
        # return self.gamma_denoising, self.ft_denoising_steps, denoising_inds

        discount = tf.map_fn(
            lambda i: self.gamma_denoising ** (self.ft_denoising_steps -tf.cast(i, tf.float32) - 1),
            denoising_inds,
            dtype=tf.float32
        )

        advantages = advantages * discount
        # return newlogprobs
        logratio = newlogprobs - oldlogprobs
        ratio = tf.exp(logratio)

        t = tf.cast(denoising_inds, tf.float32) / (self.ft_denoising_steps - 1)
        if self.ft_denoising_steps > 1:
            clip_ploss_coef = self.clip_ploss_coef_base + (
                self.clip_ploss_coef - self.clip_ploss_coef_base
            ) * (tf.exp(self.clip_ploss_coef_rate * t) - 1) / (
                math.exp(self.clip_ploss_coef_rate) - 1
            )
        else:
            clip_ploss_coef = t

        approx_kl = tf.reduce_mean((ratio - 1) - logratio)
        clipfrac = tf.reduce_mean(tf.cast(tf.abs(ratio - 1.0) > clip_ploss_coef, tf.float32))

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * tf.clip_by_value(ratio, 1 - clip_ploss_coef, 1 + clip_ploss_coef)
        pg_loss = tf.reduce_mean(tf.maximum(pg_loss1, pg_loss2))

        newvalues = self.critic(obs)
        if self.clip_vloss_coef is not None:
            v_loss_unclipped = tf.square(newvalues - returns)
            v_clipped = oldvalues + tf.clip_by_value(
                newvalues - oldvalues,
                -self.clip_vloss_coef,
                self.clip_vloss_coef,
            )
            v_loss_clipped = tf.square(v_clipped - returns)
            v_loss_max = tf.maximum(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * tf.reduce_mean(v_loss_max)
        else:
            v_loss = 0.5 * tf.reduce_mean(tf.square(newvalues - returns))

        return (
            pg_loss,
            entropy_loss,
            v_loss,
            clipfrac,
            approx_kl,
            tf.reduce_mean(ratio),
            bc_loss,
            tf.reduce_mean(eta),
        )