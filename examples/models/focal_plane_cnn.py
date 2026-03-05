"""
Focal-plane CNN actor-critic model for CleanRL PPO.

A shared convolutional trunk extracts features from focal-plane images,
then separate dense heads produce critic values and actor (mean + logstd) outputs.

This architecture is designed for telescope adaptive optics control where
observations are 2D focal-plane images (e.g. PSFs).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from dataclasses import dataclass


@dataclass
class ModelConfig:
    conv_layers: int = 3
    conv_kernel_size: int = 5
    conv_channels: int = 128
    dense_layers: int = 3
    dense_size: int = 128
    init_act_std: float = 0.1
    init_logstd_bias: float = -2.5


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """
    CNN actor-critic agent for focal-plane wavefront sensing.

    Args:
        obs_shape: Observation shape, e.g. (1, 128, 128)
        action_shape: Action shape, e.g. (45,)
        model_config: ModelConfig with architecture hyperparameters
    """

    def __init__(self, obs_shape, action_shape, model_config=None):
        super().__init__()

        if model_config is None:
            model_config = ModelConfig()

        n_out = np.prod(action_shape)

        # Build shared convolutional trunk
        conv_layers = []
        conv_layers += [layer_init(nn.Conv2d(
            obs_shape[0],
            model_config.conv_channels,
            model_config.conv_kernel_size,
            stride=2,
            padding='valid'
        ))]
        conv_layers += [nn.ReLU()]
        for _ in range(model_config.conv_layers - 1):
            conv_layers += [layer_init(nn.Conv2d(
                model_config.conv_channels,
                model_config.conv_channels,
                model_config.conv_kernel_size,
                stride=2,
                padding='valid'
            ))]
            conv_layers += [nn.ReLU()]
        conv_layers += [nn.Flatten()]

        self.conv = nn.Sequential(*conv_layers)

        # Determine flattened conv output size
        test_x = torch.zeros((1,) + obs_shape)
        conv_flat_size = self.conv(test_x).shape[1]

        # Critic head
        c_layers = [self.conv]
        c_layers += [layer_init(nn.Linear(conv_flat_size, model_config.dense_size))]
        c_layers += [nn.ReLU()]
        for _ in range(model_config.dense_layers - 1):
            c_layers += [layer_init(nn.Linear(model_config.dense_size, model_config.dense_size))]
            c_layers += [nn.ReLU()]
        c_layers += [layer_init(nn.Linear(model_config.dense_size, 1), std=1.0)]
        self.critic = nn.Sequential(*c_layers)

        # Actor head (shared pre-network, separate mean and logstd outputs)
        a_layers = [self.conv]
        a_layers += [layer_init(nn.Linear(conv_flat_size, model_config.dense_size))]
        a_layers += [nn.ReLU()]
        for _ in range(model_config.dense_layers - 1):
            a_layers += [layer_init(nn.Linear(model_config.dense_size, model_config.dense_size))]
            a_layers += [nn.ReLU()]
        self.actor_pre = nn.Sequential(*a_layers)

        init_std = model_config.init_act_std
        self.action_out = layer_init(nn.Linear(model_config.dense_size, n_out), std=init_std)
        self.logstd_out = layer_init(
            nn.Linear(model_config.dense_size, n_out),
            bias_const=model_config.init_logstd_bias,
            std=init_std,
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        actor_pre_x = self.actor_pre(x)
        action_mean = self.action_out(actor_pre_x)
        action_logstd = self.logstd_out(actor_pre_x)

        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
