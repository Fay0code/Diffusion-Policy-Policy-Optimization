import copy
import tensorflow as tf
import logging
import tensorflow_probability as tfp
import numpy as np

log = logging.getLogger(__name__)

from diffusion import DiffusionModel, Sample
from sampling import make_timesteps, extract

tfd = tfp.distributions

class VPGDiffusion(DiffusionModel):
    """
    定义完整的diffusion和RL模型
    """

    def __init__(
        self,
        actor,
        critic,
        ft_denoising_steps,
        ft_denoising_steps_d=0,
        ft_denoising_steps_t=0,
        network_path=None,
        min_sampling_denoising_std=0.1,
        min_logprob_denoising_std=0.1,
        eta=None,
        learn_eta=False,
        **kwargs,
    ):
        super().__init__(
            network=actor,
            network_path=network_path,
            **kwargs,
        )
        assert ft_denoising_steps <= self.denoising_steps
        assert ft_denoising_steps <= self.ddim_steps if self.use_ddim else True
        assert not (learn_eta and not self.use_ddim), "Cannot learn eta with DDPM."

        self.ft_denoising_steps = ft_denoising_steps
        self.ft_denoising_steps_d = ft_denoising_steps_d
        self.ft_denoising_steps_t = ft_denoising_steps_t
        self.ft_denoising_steps_cnt = 0

        self.min_sampling_denoising_std = min_sampling_denoising_std
        self.min_logprob_denoising_std = min_logprob_denoising_std
        self.x = kwargs.get("x")
        self.time = kwargs.get("time")
        self.cond = kwargs.get("cond")


        self.learn_eta = learn_eta
        if eta is not None:
            self.eta = eta
            if not learn_eta:
                self.eta.trainable = False
                logging.info("Turned off gradients for eta")

        self.actor = self.network
        self.actor_ft = tf.keras.models.clone_model(self.actor)
        self.actor_ft(self.x, self.time, self.cond)
        self.actor_ft.set_weights(self.actor.get_weights())

        logging.info("Cloned model for fine-tuning")

        self.actor.trainable = False
        logging.info("Turned off gradients of the pretrained network")
        logging.info(
            f"Number of finetuned parameters: {sum([tf.size(p).numpy() for p in self.actor_ft.trainable_variables])}"
        )

        self.critic = critic
        if network_path is not None:
            checkpoint = tf.train.Checkpoint(model=self)
            checkpoint.restore(network_path).expect_partial()
            logging.info("Loaded critic from %s", network_path)

    def step(self):
        """
        Anneal min_sampling_denoising_std and fine-tuning denoising steps

        Current configs do not apply annealing
        """
        if isinstance(self.min_sampling_denoising_std, tf.Variable):
            self.min_sampling_denoising_std.assign_sub(self.ft_denoising_steps_d)

        self.ft_denoising_steps_cnt += 1
        if (
            self.ft_denoising_steps_d > 0
            and self.ft_denoising_steps_t > 0
            and self.ft_denoising_steps_cnt % self.ft_denoising_steps_t == 0
        ):
            self.ft_denoising_steps = max(
                0, self.ft_denoising_steps - self.ft_denoising_steps_d
            )

            self.actor = self.network
            self.actor_ft = tf.keras.models.clone_model(self.actor)
            self.actor_ft(self.x, self.time, self.cond)
            self.actor_ft.set_weights(self.actor.get_weights())
            self.actor.trainable = False
            logging.info(
                f"Finished annealing fine-tuning denoising steps to {self.ft_denoising_steps}"
            )

    def get_min_sampling_denoising_std(self):
        if isinstance(self.min_sampling_denoising_std, float):
            return self.min_sampling_denoising_std
        else:
            return self.min_sampling_denoising_std.numpy()

    def p_mean_var(
        self,
        x,
        t,
        cond,
        index=None,
        use_base_policy=False,
        deterministic=False,
    ):
        
        noise = self.actor(x, t, cond=cond)
        if self.use_ddim:
            ft_indices = tf.where(index >= (self.ddim_steps - self.ft_denoising_steps))
            ft_indices = tf.squeeze(ft_indices, axis=-1)
        else:
            ft_indices = tf.where(t < self.ft_denoising_steps)
            ft_indices = tf.squeeze(ft_indices, axis=-1)

        actor = self.actor if use_base_policy else self.actor_ft
        # tf.print("ft_indices:",ft_indices)
        
        actor = self.actor if use_base_policy else self.actor_ft
    
        if tf.size(ft_indices) > 0:
            cond_ft = {key: tf.gather(cond[key], ft_indices) for key in cond}
            noise_ft = actor(tf.gather(x, ft_indices), tf.gather(t, ft_indices), cond=cond_ft)
            noise = tf.tensor_scatter_nd_update(noise, tf.expand_dims(ft_indices, axis=-1), noise_ft)
        
        # return a

        if self.predict_epsilon:
            if self.use_ddim:
                alpha = extract(self.ddim_alphas, index, x.shape)
                alpha_prev = extract(self.ddim_alphas_prev, index, x.shape)
                sqrt_one_minus_alpha = extract(self.ddim_sqrt_one_minus_alphas, index, x.shape)
                x_recon = (x - sqrt_one_minus_alpha * noise) / tf.sqrt(alpha)
            else:
                x_recon = (
                    extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
                    - extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * noise
                )
        else:
            x_recon = noise

        if self.denoised_clip_value is not None:
            x_recon = tf.clip_by_value(x_recon, clip_value_min=-self.denoised_clip_value, clip_value_max=self.denoised_clip_value)
            if self.use_ddim:
                noise = (x - tf.sqrt(alpha) * x_recon) / sqrt_one_minus_alpha

        if self.use_ddim and self.eps_clip_value is not None:
            noise = tf.clip_by_value(noise, -self.eps_clip_value, self.eps_clip_value)

        if self.use_ddim:
            if deterministic:
                etas = tf.zeros((x.shape[0], 1, 1))
            else:
                etas = self.eta(cond)
            sigma = (
                etas
                * tf.sqrt((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev))
            )
            dir_xt_coef = tf.sqrt(tf.clip_by_value(1.0 - alpha_prev - sigma**2, 0, np.inf))
            mu = tf.sqrt(alpha_prev) * x_recon + dir_xt_coef * noise
            var = sigma**2
            logvar = tf.math.log(var)
        else:
            mu = (
                extract(self.ddpm_mu_coef1, t, x.shape) * x_recon
                + extract(self.ddpm_mu_coef2, t, x.shape) * x
            )
            logvar = extract(self.ddpm_logvar_clipped, t, x.shape)
            etas = tf.ones_like(mu)
        return mu, logvar, etas
        # return ft_indices

    @tf.function
    def call(
        self,
        cond,
        deterministic=False,
        return_chain=True,
        use_base_policy=False,
    ):
        sample_data = cond["state"] if "state" in cond else cond["rgb"]
        B = len(sample_data)

        min_sampling_denoising_std = self.get_min_sampling_denoising_std()

        x = tf.random.normal((B, self.horizon_steps, self.action_dim))
        if self.use_ddim:
            t_all = self.ddim_t
        else:
            t_all = list(reversed(range(self.denoising_steps)))
        chain = [] if return_chain else None
        if not self.use_ddim and self.ft_denoising_steps == self.denoising_steps:
            chain.append(x)
        if self.use_ddim and self.ft_denoising_steps == self.ddim_steps:
            chain.append(x)
        for i, t in enumerate(t_all):
            t_b = make_timesteps(B, t, self.device)
            index_b = make_timesteps(B, i, self.device)
            mean, logvar, _ = self.p_mean_var(x,t_b,cond, index_b)
            # r = self.p_mean_var(x,t_b,cond, index_b)
            # return x,t_b,cond, index_b
            # return r
        
            std = tf.exp(0.5 * logvar)

            if self.use_ddim:
                if deterministic:
                    std = tf.zeros_like(std)
                else:
                    std = tf.clip_by_value(std, clip_value_min=min_sampling_denoising_std, clip_value_max=np.inf)
            else:
                if deterministic and t == 0:
                    std = tf.zeros_like(std)
                elif deterministic:
                    std = tf.clip_by_value(std, clip_value_min=1e-3, clip_value_max=np.inf)
                else:
                    std = tf.clip_by_value(std, clip_value_min=min_sampling_denoising_std,clip_value_max=np.inf)
            noise = tf.random.normal(x.shape)
            noise = tf.clip_by_value(noise, -self.randn_clip_value, self.randn_clip_value)
            x = mean + std * noise

            # return noise

            if self.final_action_clip_value is not None and i == len(t_all) - 1:
                x = tf.clip_by_value(x, -self.final_action_clip_value, self.final_action_clip_value)

            if return_chain:
                if not self.use_ddim and t <= self.ft_denoising_steps:
                    chain.append(x)
                elif self.use_ddim and i >= (
                    self.ddim_steps - self.ft_denoising_steps - 1
                ):
                    chain.append(x)

        if return_chain:
            chain = tf.stack(chain, axis=1)
        return Sample(x, chain)

    def get_logprobs(
        self,
        cond,
        chains,
        get_ent: bool = False,
        use_base_policy: bool = False,
    ):
        cond = {
            key: tf.repeat(cond[key], repeats=self.ft_denoising_steps, axis=0)
            for key in cond
        }

        if self.use_ddim:
            t_single = self.ddim_t[-self.ft_denoising_steps :]
        else:
            t_single = tf.range(
                start=self.ft_denoising_steps - 1,
                limit=-1,
                delta=-1,
                dtype=tf.int32,
            )
        t_all = tf.repeat(t_single, repeats=chains.shape[0])
        if self.use_ddim:
            indices_single = tf.range(
                start=self.ddim_steps - self.ft_denoising_steps,
                limit=self.ddim_steps,
                dtype=tf.int32,
            )
            indices = tf.repeat(indices_single, repeats=chains.shape[0])
        else:
            indices = None

        chains_prev = chains[:, :-1]
        chains_next = chains[:, 1:]

        chains_prev = tf.reshape(chains_prev, (-1, self.horizon_steps, self.action_dim))
        chains_next = tf.reshape(chains_next, (-1, self.horizon_steps, self.action_dim))

        next_mean, logvar, eta = self.p_mean_var(
            chains_prev,
            t_all,
            cond=cond,
            index=indices,
            use_base_policy=use_base_policy,
        )
        std = tf.exp(0.5 * logvar)
        std = tf.clip_by_value(std, clip_value_min=self.min_logprob_denoising_std,clip_value_max=np.inf)
        dist = tfd.Normal(loc=next_mean, scale=std)

        log_prob = dist.log_prob(chains_next)
        if get_ent:
            return log_prob, eta
        return log_prob

    def get_logprobs_subsample(
        self,
        cond,
        chains_prev,
        chains_next,
        denoising_inds,
        get_ent: bool = False,
        use_base_policy: bool = False,
    ):
        if self.use_ddim:
            t_single = self.ddim_t[-self.ft_denoising_steps :]
        else:
            t_single = tf.range(
                start=self.ft_denoising_steps - 1,
                limit=-1,
                delta=-1,
                dtype=tf.int32,
            )
        t_all = tf.gather(t_single, denoising_inds)
        
        if self.use_ddim:
            ddim_indices_single = tf.range(
                start=self.ddim_steps - self.ft_denoising_steps,
                limit=self.ddim_steps,
                dtype=tf.int32,
            )
            ddim_indices = tf.gather(ddim_indices_single, denoising_inds)
        else:
            ddim_indices = None
        # return chains_prev, t_all, cond, ddim_indices
    
        next_mean, logvar, eta = self.p_mean_var(
            chains_prev,
            t_all,
            cond=cond,
            index=ddim_indices,
            use_base_policy=use_base_policy,
        )
        std = tf.exp(0.5 * logvar)
        std = tf.clip_by_value(std, clip_value_min=self.min_logprob_denoising_std,clip_value_max=np.inf)
        dist = tfd.Normal(loc=next_mean, scale=std)

        log_prob = dist.log_prob(chains_next)
        if get_ent:
            return log_prob, eta
        return log_prob

    # def loss(self, cond, chains, reward):
    #     with tf.GradientTape() as tape:
    #         value = self.critic(cond)
    #         advantage = reward - tf.squeeze(value)

    #         logprobs, eta = self.get_logprobs(cond, chains, get_ent=True)

    #         logprobs = tf.reduce_sum(logprobs[:, :, : self.action_dim], axis=-1)

    #         logprobs = tf.reshape(logprobs, (-1, self.denoising_steps, self.horizon_steps))

    #         logprobs = tf.reduce_mean(logprobs, axis=-2)

    #         logprobs = tf.reduce_mean(logprobs, axis=-1)

    #         loss_actor = tf.reduce_mean(-logprobs * advantage)

    #         pred = self.critic(cond)
    #         loss_critic = tf.reduce_mean(tf.square(tf.squeeze(pred) - reward))

    #     grads = tape.gradient(loss_actor, self.actor_ft.trainable_variables)
    #     self.optimizer.apply_gradients(zip(grads, self.actor_ft.trainable_variables))

    #     return loss_actor, loss_critic, eta

# import tensorflow as tf
# import copy

# # 定义一个简单的模型
# class DummyNetwork(tf.keras.Model):
#     def __init__(self):
#         super(DummyNetwork, self).__init__()
#         self.dense = tf.keras.layers.Dense(10)

#     def call(self, x):
#         return self.dense(x)

# # 创建原始模型实例
# original_model = DummyNetwork()

# # 构建模型
# x = tf.random.normal((1, 10))
# original_model(x)

# # 复制模型
# cloned_model = tf.keras.models.clone_model(original_model)
# cloned_model.build((None, 10))

# # 复制权重
# cloned_model.set_weights(original_model.get_weights())
# cloned_model.count_params()

# # 验证复制是否成功
# print("Original model weights:", original_model.get_weights())
# print("Cloned model weights:", cloned_model.get_weights())
# print("Number of weights:", cloned_model.count_params())