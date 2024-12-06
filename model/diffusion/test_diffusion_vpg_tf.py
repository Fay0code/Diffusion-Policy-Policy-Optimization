import unittest
import tensorflow as tf
from diffusion_vpg_tf import VPGDiffusion
from model.diffusion.diffusion import DiffusionModel
from mlp_diffusion_tf import DiffusionMLP
from model.common.critic_tf import CriticObs
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,7'

class TestVPGDiffusion(unittest.TestCase):
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
            use_layernorm=False
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
    

        self.vpg_diffusion = VPGDiffusion(
        actor=self.actor,
        critic=self.critic,
        ft_denoising_steps=10,
        horizon_steps=4,
        obs_dim=11,
        action_dim=3,
        denoising_steps=20,
        # use_ddim=False,
        # ddim_steps=50,
        # predict_epsilon=True,
        # denoised_clip_value=1.0,
        # randn_clip_value=10,
        # final_action_clip_value=None,
        # eps_clip_value=None,
        x=x,
        time=time,
        cond=cond,
    )

        self.vpg_diffusion.actor_ft(x, time, cond)
    

    # def test_step(self):
    #     initial_steps = self.vpg_diffusion.ft_denoising_steps
    #     self.vpg_diffusion.step()
    #     self.assertEqual(self.vpg_diffusion.ft_denoising_steps, initial_steps)

    # def test_get_min_sampling_denoising_std(self):
    #     std = self.vpg_diffusion.get_min_sampling_denoising_std()
    #     self.assertEqual(std, 0.1)

    # def test_p_mean_var(self):
    #     x = tf.random.normal((2, 4, 3))
    #     t = tf.constant([19,19], dtype=tf.int32)
    #     cond = {'state': tf.random.normal((2, 4, 11))}
    #     index = tf.constant([0, 1], dtype=tf.int32)
    #     r=self.vpg_diffusion.p_mean_var(x, t, cond, index) 
    #     print("r:", r)  
        # # mu, logvar, etas = self.vpg_diffusion.p_mean_var(x, t, cond, index)   
        # self.assertEqual(mu.shape, x.shape)
        # self.assertEqual(logvar.shape, (2,1,1))
        # self.assertEqual(etas.shape, x.shape)

    # def test_call(self):
    #     cond = {'state': tf.random.normal((2, 4, 11))}
    #     sample = self.vpg_diffusion(cond)
    #     self.assertEqual(sample.trajectories.shape, (2, 4, 3))

    # def test_get_logprobs(self):
    #     cond = {'state': tf.random.normal((2, 4, 11))}
    #     chains = tf.random.normal((2, 11, 4, 3))
    #     logprobs = self.vpg_diffusion.get_logprobs(cond, chains)
    #     self.assertEqual(logprobs.shape, (2 * 10, 4, 3))

    def test_get_logprobs_subsample(self):
        cond = {'state': tf.random.normal((2, 4, 11))}
        chains_prev = tf.random.normal((2, 4, 3))
        chains_next = tf.random.normal((2, 4, 3))
        denoising_inds = tf.constant([0, 1], dtype=tf.int32)
        logprobs = self.vpg_diffusion.get_logprobs_subsample(cond, chains_prev, chains_next, denoising_inds)
        # print("logprobs:", logprobs)
        self.assertEqual(logprobs.shape, (2, 4, 3))

    # def test_loss(self):
    #     cond = {'state': tf.random.normal((2, 4, 11))}
    #     chains = tf.random.normal((2, 11, 4, 3))
    #     reward = tf.random.normal((2,))
    #     loss_actor, loss_critic, eta = self.vpg_diffusion.loss(cond, chains, reward)
    #     self.assertIsInstance(loss_actor, tf.Tensor)
    #     self.assertIsInstance(loss_critic, tf.Tensor)
    #     self.assertIsInstance(eta, tf.Tensor)

if __name__ == '__main__':
    unittest.main()