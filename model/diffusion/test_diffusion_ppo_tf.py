import unittest
import tensorflow as tf
from diffusion_ppo_tf import PPODiffusion
from mlp_diffusion_tf import DiffusionMLP
from model.common.critic_tf import CriticObs

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,6'

class TestPPODiffusion(unittest.TestCase):
    def setUp(self):
        self.actor = DiffusionMLP(
            action_dim=3,
            horizon_steps=4,
            cond_dim=11,
            time_dim=16,
            mlp_dims=[512, 512, 512],
            cond_mlp_dims=None,
            activation_type="ReLU",
            out_activation_type="Identity",
            use_layernorm=False,
            residual_style=True
        )
        
        self.critic = CriticObs(
            obs_dim=11,
            cond_dim=11,
            mlp_dims=[256, 256, 256],
            activation_type="Mish",
            out_activation_type="Identity",
            use_layernorm=False,
            residual_style=True,
        )
        # Call the model with real tensor data to build it
        x = tf.random.normal((2, 4, 3))
        time = tf.constant([1,2], dtype=tf.float32)
        cond = {
            "state": tf.random.normal((2, 4, 11)),
            "rgb": tf.random.normal((2, 4, 3, 32, 32))
        }
        self.actor(x, time, cond)
        self.critic(cond)
        
        self.ppo_diffusion = PPODiffusion(
            actor=self.actor,
            critic=self.critic,
            ft_denoising_steps=10,
            horizon_steps=4,
            obs_dim=11,
            action_dim=3,
            gamma_denoising=0.99,
            clip_ploss_coef=0.2,
            clip_ploss_coef_base=1e-3,
            clip_ploss_coef_rate=3,
            x=x,
            time=time,
            cond=cond,
        )

    # def test_initialization(self):
    #     self.assertIsInstance(self.ppo_diffusion, VPGDiffusion)
    #     self.assertIsInstance(self.ppo_diffusion, DiffusionModel)

    # def test_p_mean_var(self):
    #     x = tf.random.normal((2, 4, 3))
    #     t = tf.constant([1,5])
    #     cond = {"state": tf.random.normal((2, 4, 11))}
    #     mu, logvar,_ = self.ppo_diffusion.p_mean_var(x, t, cond)
    #     self.assertEqual(mu.shape, x.shape)
    #     self.assertEqual(logvar.shape, [2,1,1])

    # def test_call(self):
    #     cond = {"state": tf.random.normal((2, 4, 11))}
    #     sample = self.ppo_diffusion(cond)
    #     self.assertEqual(sample.trajectories.shape, (2, 4, 3))

    def test_loss(self):
        obs = {"state": tf.random.normal((2, 4, 11))}
        chains_prev = tf.random.normal((2, 4, 3))
        chains_next = tf.random.normal((2, 4, 3))
        denoising_inds = tf.constant([0, 1], dtype=tf.int32)
        returns = tf.random.normal((2,))
        oldvalues = tf.random.normal((2,))
        advantages = tf.random.normal((2,))
        oldlogprobs = tf.random.normal((2,4,3))

        r= self.ppo_diffusion.loss(
            obs,
            chains_prev,
            chains_next,
            denoising_inds,
            returns,
            oldvalues,
            advantages,
            oldlogprobs,
            use_bc_loss=True,
            reward_horizon=4,
        )
        print(r)
        
        # pg_loss, entropy_loss, v_loss, clipfrac, approx_kl, ratio_mean, bc_loss, eta_mean = self.ppo_diffusion.loss(
        #     obs,
        #     chains_prev,
        #     chains_next,
        #     denoising_inds,
        #     returns,
        #     oldvalues,
        #     advantages,
        #     oldlogprobs,
        #     use_bc_loss=True,
        #     reward_horizon=4,
        # )


        # 打印损失值
        # tf.print("pg_loss:", pg_loss)
        # tf.print("entropy_loss:", entropy_loss)
        # tf.print("v_loss:", v_loss)
        # tf.print("clipfrac:", clipfrac)
        # tf.print("approx_kl:", approx_kl)
        # tf.print("ratio_mean:", ratio_mean)
        # tf.print("bc_loss:", bc_loss)
        # tf.print("eta_mean:", eta_mean)

        # self.assertIsInstance(pg_loss, tf.Tensor)
        # self.assertIsInstance(entropy_loss, tf.Tensor)
        # self.assertIsInstance(v_loss, tf.Tensor)
        # self.assertIsInstance(clipfrac, tf.Tensor)
        # self.assertIsInstance(approx_kl, tf.Tensor)
        # self.assertIsInstance(ratio_mean, tf.Tensor)
        # self.assertIsInstance(bc_loss, tf.Tensor)
        # self.assertIsInstance(eta_mean, tf.Tensor)

if __name__ == '__main__':
    unittest.main()

    # def test_discount(self):

    #     r1 = tf.constant(0.99, dtype=tf.float32)
    #     r2 = tf.constant(10, dtype=tf.float32)
    #     r3 = tf.constant([0, 1, 2], dtype=tf.int32)
    #     r4 = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)

    #     discount = tf.map_fn(
    #         lambda i: r1 ** (r2 - tf.cast(i, tf.float32) - 1),
    #         r3,
    #         dtype=tf.float32
    #     )

    #     r4 *= discount
    #     tf.print("discounted advantages:", r4)