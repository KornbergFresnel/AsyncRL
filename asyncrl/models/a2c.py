from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn

from asyncrl.models.dqn import SimplePreprocessor


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find("Linear") != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class ActorCritic(torch.nn.Module):
    def __init__(self, observation_space, action_space):
        super(ActorCritic, self).__init__()

        self.observation_space = observation_space
        self.action_space = action_space

        self.preprocessor_actor = self.create_preprocessor(256)
        self.preprocessor_critic = self.create_preprocessor(256)
        self.actor_linear = self.create_actor(256)
        self.critic_linear = self.create_critic(256)

        self.apply(weights_init)

        # self.lstm.bias_ih.data.fill_(0)
        # self.lstm.bias_hh.data.fill_(0)

        self.train()

    def create_preprocessor(self, num_outputs: int) -> nn.Module:
        return SimplePreprocessor(self.observation_space.shape[0], num_outputs)

    def create_critic(self, num_inputs: int) -> nn.Module:
        critic_linear = nn.Linear(num_inputs, 1)
        critic_linear.weight.data = normalized_columns_initializer(
            critic_linear.weight.data, 1.0
        )
        critic_linear.bias.data.fill_(0)
        return critic_linear

    def create_actor(self, num_inputs: int) -> nn.Module:
        actor_linear = nn.Linear(num_inputs, self.action_space.n)
        actor_linear.weight.data = normalized_columns_initializer(
            actor_linear.weight.data, 0.01
        )
        actor_linear.bias.data.fill_(0)
        return actor_linear

    def forward(self, obs: Any, state: Tuple[torch.Tensor, torch.Tensor]):
        x, state = self.preprocessor_actor(obs, state)
        x_critic, _ = self.preprocessor_critic(obs, state)
        return self.critic_linear(x_critic), self.actor_linear(x), state

    def init_state(self, batch_size: int, device=None):
        return self.preprocessor_actor.init_state(batch_size, device)
