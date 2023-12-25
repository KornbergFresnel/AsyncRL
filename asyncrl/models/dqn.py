from typing import Tuple, Any

import torch
import torch.nn as nn

from gym import spaces


class Preprocessor(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.conv = nn.Sequential(
            nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ELU(),
        )

        self.lstm = nn.LSTMCell(32 * 3 * 3, num_outputs)

    def forward(
        self, obs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.conv(obs)
        x = x.view(-1, 32 * 3 * 3)
        hx, cx = self.lstm(x, states)
        return hx, (hx, cx)

    def init_state(
        self, batch_size: int, device: torch.DeviceObjType = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if device is None:
            device = next(self.parameters()).device
        hx = torch.zeros(batch_size, 256).to(device)
        cx = torch.zeros(batch_size, 256).to(device)
        return (hx, cx)


class SimplePreprocessor(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(num_inputs, 256),
            nn.ReLU(),
            nn.Linear(256, num_outputs),
            nn.ReLU(),
        )

    def forward(
        self, obs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.embed(obs)
        return x, states

    def init_state(
        self, batch_size: int, device: torch.DeviceObjType = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if device is None:
            device = next(self.parameters()).device
        hx = torch.zeros(batch_size, 2).to(device)
        cx = torch.zeros(batch_size, 2).to(device)
        return (hx, cx)


class DQN(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        state_embedding_size: int = 256,
    ) -> None:
        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space

        self.preprocessor = self.create_preprocessor(state_embedding_size)
        self.q: nn.Module = self.create_critic(state_embedding_size)

    def create_preprocessor(self, num_outputs: int) -> nn.Module:
        return SimplePreprocessor(self.observation_space.shape[0], num_outputs)

    def create_critic(self, num_inputs: int) -> nn.Module:
        # default is for atari
        return nn.Sequential(
            nn.Linear(num_inputs, num_inputs),
            nn.ReLU(),
            nn.Linear(num_inputs, self.action_space.n),
        )

    def forward(
        self, obs: torch.Tensor, states: Tuple[Any, Any] = None
    ) -> Tuple[torch.Tensor, Tuple[Any, Any]]:
        state_embedding, states = self.preprocessor(obs, states)

        q = self.q(state_embedding)

        return q, states

    def init_state(self, batch_size: int, device=None):
        return self.preprocessor.init_state(batch_size, device)
