"""
generate a sequence of actions by sampling from the diffusion model
add noise to the sampled actions
generate data by denoising the noisy actions

DDPM:need more denoise steps to get better results

DDIM: first compute x0, then compute the denoised data, use less steps
"""


import tensorflow as tf
import numpy as np
import logging
from collections import namedtuple
from sampling import (
    extract,
    cosine_beta_schedule,
    make_timesteps,
)

log = logging.getLogger(__name__)

Sample = namedtuple("Sample", "trajectories chains")

class DiffusionModel(tf.keras.Model):

    def __init__(
        self,
        network,
        horizon_steps,
        obs_dim,
        action_dim,
        network_path=None,
        device="/GPU:0",
        # Various clipping
        denoised_clip_value=1.0,
        randn_clip_value=10,
        final_action_clip_value=None,
        eps_clip_value=None,  # DDIM only
        # DDPM parameters
        denoising_steps=20,
        predict_epsilon=True,
        # DDIM sampling
        use_ddim=False,
        ddim_discretize="uniform",
        ddim_steps=None,
        **kwargs,
    ):
        super(DiffusionModel, self).__init__()
        self.device = device
        self.horizon_steps = horizon_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.denoising_steps = int(denoising_steps)
        self.predict_epsilon = predict_epsilon
        self.use_ddim = use_ddim
        self.ddim_steps = ddim_steps

        # Clip noise value at each denoising step
        self.denoised_clip_value = denoised_clip_value

        # Whether to clamp the final sampled action between [-1, 1]
        self.final_action_clip_value = final_action_clip_value

        # For each denoising step, we clip sampled randn (from standard deviation) such that the sampled action is not too far away from mean
        self.randn_clip_value = randn_clip_value

        # Clip epsilon for numerical stability
        self.eps_clip_value = eps_clip_value

        # Set up models
        with tf.device(device):
            self.network = network
        if network_path is not None:
            checkpoint = tf.train.Checkpoint(model=self.network)
            checkpoint.restore(network_path).expect_partial()
            if "ema" in checkpoint:
                logging.info("Loaded SL-trained policy from %s", network_path)
            else:
                logging.info("Loaded RL-trained policy from %s", network_path)
        logging.info(
            f"Number of network parameters: {self.network.count_params()}"
        )

        """
        DDPM parameters

        """
        """
        βₜ
        """
        self.betas = cosine_beta_schedule(denoising_steps)
        """
        αₜ = 1 - βₜ
        """
        self.alphas = 1.0 - self.betas
        """
        α̅ₜ= ∏ᵗₛ₌₁ αₛ 
        """
        self.alphas_cumprod = tf.math.cumprod(self.alphas, axis=0)
        """
        α̅ₜ₋₁
        """
        self.alphas_cumprod_prev = tf.concat(
            [tf.ones(1), self.alphas_cumprod[:-1]], axis=0
        )
        """
        √ α̅ₜ
        """
        self.sqrt_alphas_cumprod = tf.sqrt(self.alphas_cumprod)
        """
        √ 1-α̅ₜ
        """
        self.sqrt_one_minus_alphas_cumprod = tf.sqrt(1.0 - self.alphas_cumprod)
        """
        √ 1\α̅ₜ
        """
        self.sqrt_recip_alphas_cumprod = tf.sqrt(1.0 / self.alphas_cumprod)
        """
        √ 1\α̅ₜ-1
        """
        self.sqrt_recipm1_alphas_cumprod = tf.sqrt(1.0 / self.alphas_cumprod - 1)
        """
        β̃ₜ = σₜ² = βₜ (1-α̅ₜ₋₁)/(1-α̅ₜ)
        """
        self.ddpm_var = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.ddpm_logvar_clipped = tf.math.log(tf.clip_by_value(self.ddpm_var, 1e-20, np.inf))
        """
        μₜ = β̃ₜ √ α̅ₜ₋₁/(1-α̅ₜ)x₀ + √ αₜ (1-α̅ₜ₋₁)/(1-α̅ₜ)xₜ
        """
        self.ddpm_mu_coef1 = (
            self.betas
            * tf.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.ddpm_mu_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * tf.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

        """
        DDIM parameters

        """
        if use_ddim:
            assert predict_epsilon, "DDIM requires predicting epsilon for now."
            if ddim_discretize == "uniform":  # use the HF "leading" style
                step_ratio = self.denoising_steps // ddim_steps
                self.ddim_t = (
                    tf.range(0, ddim_steps) * step_ratio
                )
            else:
                raise ValueError("Unknown discretization method for DDIM.")
            self.ddim_alphas = (
                tf.gather(self.alphas_cumprod, self.ddim_t)
            )
            self.ddim_alphas_sqrt = tf.sqrt(self.ddim_alphas)
            self.ddim_alphas_prev = tf.concat(
                [
                    tf.constant([1.0], dtype=tf.float32),
                    tf.gather(self.alphas_cumprod, self.ddim_t[:-1]),
                ],
                axis=0
            )
            self.ddim_sqrt_one_minus_alphas = tf.sqrt(1.0 - self.ddim_alphas)

            # Initialize fixed sigmas for inference - eta=0
            ddim_eta = 0
            self.ddim_sigmas = (
                ddim_eta
                * tf.sqrt(
                    (1 - self.ddim_alphas_prev)
                    / (1 - self.ddim_alphas)
                    * (1 - self.ddim_alphas / self.ddim_alphas_prev)
                )
            )

            # Flip all
            self.ddim_t = tf.reverse(self.ddim_t, axis=[0])
            self.ddim_alphas = tf.reverse(self.ddim_alphas, axis=[0])
            self.ddim_alphas_sqrt = tf.reverse(self.ddim_alphas_sqrt, axis=[0])
            self.ddim_alphas_prev = tf.reverse(self.ddim_alphas_prev, axis=[0])
            self.ddim_sqrt_one_minus_alphas = tf.reverse(
                self.ddim_sqrt_one_minus_alphas, axis=[0]
            )
            self.ddim_sigmas = tf.reverse(self.ddim_sigmas, axis=[0])

    # ---------- Sampling ----------#
    #compute mean and variance of the predicted distribution
    def p_mean_var(self, x, t, cond, index=None, network_override=None):
        if network_override is not None:
            noise = network_override(x, t, cond=cond)
        else:
            noise = self.network(x, t, cond=cond)

        # Predict x_0
        if self.predict_epsilon:
            if self.use_ddim:
                """
                x₀ = (xₜ - √ (1-αₜ) ε )/ √ αₜ
                """
                alpha = extract(self.ddim_alphas, index, x.shape)
                alpha_prev = extract(self.ddim_alphas_prev, index, x.shape)
                sqrt_one_minus_alpha = extract(
                    self.ddim_sqrt_one_minus_alphas, index, x.shape
                )
                x_recon = (x - sqrt_one_minus_alpha * noise) / tf.sqrt(alpha)
            else:
                """
                x₀ = √ 1\α̅ₜ xₜ - √ 1\α̅ₜ-1 ε
                """
                x_recon = (
                    extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
                    - extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * noise
                )
        else:  # directly predicting x₀
            x_recon = noise
        if self.denoised_clip_value is not None:
            x_recon = tf.clip_by_value(x_recon, -self.denoised_clip_value, self.denoised_clip_value)
            if self.use_ddim:
                # re-calculate noise based on clamped x_recon - default to false in HF, but let's use it here
                noise = (x - tf.sqrt(alpha) * x_recon) / sqrt_one_minus_alpha

        # Clip epsilon for numerical stability in policy gradient - not sure if this is helpful yet, but the value can be huge sometimes. This has no effect if DDPM is used
        if self.use_ddim and self.eps_clip_value is not None:
            noise = tf.clip_by_value(noise, -self.eps_clip_value, self.eps_clip_value)

        # Get mu
        if self.use_ddim:
            """
            μ = √ αₜ₋₁ x₀ + √(1-αₜ₋₁ - σₜ²) ε

            eta=0
            """
            sigma = extract(self.ddim_sigmas, index, x.shape)
            dir_xt = tf.sqrt(1.0 - alpha_prev - sigma**2) * noise
            mu = tf.sqrt(alpha_prev) * x_recon + dir_xt
            var = sigma**2
            logvar = tf.math.log(var)
        else:
            """
            μₜ = β̃ₜ √ α̅ₜ₋₁/(1-α̅ₜ)x₀ + √ αₜ (1-α̅ₜ₋₁)/(1-α̅ₜ)xₜ
            """
            mu = (
                extract(self.ddpm_mu_coef1, t, x.shape) * x_recon
                + extract(self.ddpm_mu_coef2, t, x.shape) * x
            )
            logvar = extract(self.ddpm_logvar_clipped, t, x.shape)
        return mu, logvar

    @tf.function
    #forward pass for sampling actions
    #return a sample of actions
    def call(self, cond, deterministic=True):
        """
        Forward pass for sampling actions. Used in evaluating pre-trained/fine-tuned policy. Not modifying diffusion clipping

        Args:
            cond: dict with key state/rgb; more recent obs at the end
                state: (B, To, Do)
                rgb: (B, To, C, H, W)
        Return:
            Sample: namedtuple with fields:
                trajectories: (B, Ta, Da)
        """
        sample_data = cond["state"] if "state" in cond else cond["rgb"]
        B = len(sample_data)

        # Loop
        x = tf.random.normal((B, self.horizon_steps, self.action_dim))
        if self.use_ddim:
            t_all = self.ddim_t
        else:
            t_all = list(reversed(range(self.denoising_steps)))
        for i, t in enumerate(t_all):
            t_b = make_timesteps(B, t, self.device)
            index_b = make_timesteps(B, i, self.device)
            mean, logvar = self.p_mean_var(
                x=x,
                t=t_b,
                cond=cond,
                index=index_b,
            )
            std = tf.exp(0.5 * logvar)

            # Determine noise level
            if self.use_ddim:
                std = tf.zeros_like(std)
            else:
                if t == 0:
                    std = tf.zeros_like(std)
                else:
                    std = tf.clip_by_value(std, 1e-3, np.inf)
            noise = tf.random.normal(x.shape)
            noise = tf.clip_by_value(noise, -self.randn_clip_value, self.randn_clip_value)
            x = mean + std * noise

            # clamp action at final step
            if self.final_action_clip_value is not None and i == len(t_all) - 1:
                x = tf.clip_by_value(x, -self.final_action_clip_value, self.final_action_clip_value)
        return Sample(x, None)

    # ---------- Supervised training ----------#
    
    def loss(self, x, *args):
        batch_size = len(x)
        t = tf.random.uniform(
            shape=(batch_size,), minval=0, maxval=self.denoising_steps, dtype=tf.int64
        )
        return self.p_losses(x, *args, t)

    def p_losses(
        self,
        x_start,
        cond: dict,
        t,
    ):
        """
        If predicting epsilon: E_{t, x0, ε} [||ε - ε_θ(√α̅ₜx0 + √(1-α̅ₜ)ε, t)||²

        Args:
            x_start: (batch_size, horizon_steps, action_dim)
            cond: dict with keys as step and value as observation
            t: batch of integers
        """
        # Forward process
        noise = tf.random.normal(x_start.shape)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Predict
        x_recon = self.network(x_noisy, t, cond=cond)
        if self.predict_epsilon:
            return tf.reduce_mean(tf.square(x_recon - noise))
        else:
            return tf.reduce_mean(tf.square(x_recon - x_start))

    def q_sample(self, x_start, t, noise=None):
        """
        q(xₜ | x₀) = 𝒩(xₜ; √ α̅ₜ x₀, (1-α̅ₜ)I)
        xₜ = √ α̅ₜ xₒ + √ (1-α̅ₜ) ε
        """
        if noise is None:
            noise = tf.random.normal(x_start.shape)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

