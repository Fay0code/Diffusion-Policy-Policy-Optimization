import unittest
from unittest.mock import patch, MagicMock
import tensorflow as tf
import numpy as np
import random
import os
from omegaconf import OmegaConf
import yaml
import math
from datetime import datetime


from train_agent_tf import TrainAgent


class TestTrainAgent(unittest.TestCase):

    @patch('train_agent_tf.make_async')
    @patch('model.diffusion.diffusion_ppo.PPODiffusion', autospec=True)
    def test_initialization(self, mock_ppo_diffusion, mock_make_async):
        # Register resolvers
        OmegaConf.register_new_resolver("eval", eval, replace=True)
        OmegaConf.register_new_resolver("round_up", math.ceil)
        OmegaConf.register_new_resolver("round_down", math.floor)
        OmegaConf.register_new_resolver("now", lambda pattern: datetime.now().strftime(pattern))

        # Mock configuration
        yaml_path = os.path.join(
            os.getcwd(), "dppo", "cfg", "gym", "finetune", "hopper-v2", "ft_ppo_diffusion_mlp.yaml"
        )
        with open(yaml_path, 'r') as file:
            cfg_dict = yaml.safe_load(file)

        cfg = OmegaConf.create(cfg_dict)
        # cfg.seed = 42

        # Mock the environment
        mock_env = MagicMock()
        mock_make_async.return_value = mock_env

        # Initialize TrainAgent
        agent = TrainAgent(cfg)

        # Check if the configuration is set correctly
        self.assertEqual(agent.cfg, cfg)
        self.assertEqual(agent.device, 'cuda:0')
        self.assertEqual(agent.seed, 42)

        # Check if the environment is created correctly
        mock_make_async.assert_called_once_with(
            'hopper-medium-v2',
            env_type=None,
            num_envs=40,
            asynchronous=True,
            max_episode_steps=1000,
            wrappers={
                'mujoco_locomotion_lowdim': {'normalization_path': '${normalization_path}'},
                'multi_step': {
                    'n_obs_steps': '${cond_steps}',
                    'n_action_steps': '${act_steps}',
                    'max_episode_steps': '${env.max_episode_steps}',
                    'reset_within_step': True
                }
            },
            robomimic_env_cfg_path=None,
            shape_meta=None,
            use_image_obs=False,
            render=False,
            render_offscreen=False,
            obs_dim=11,
            action_dim=3,
        )

        # Check if the environment seed is set
        if not cfg.env.get("env_type", None) == "furniture":
            mock_env.seed.assert_called_once_with([42 + i for i in range(cfg.env.n_envs)])


        # Check if the random seeds are set correctly
        self.assertEqual(np.random.get_state()[1][0], 42)

if __name__ == '__main__':
    unittest.main()