from typing import Any, Dict, Tuple
from argparse import Namespace

import numpy as np
import torch

from tensorboardX import SummaryWriter
from torch._tensor import Tensor
from torch.nn.modules import Module
from asyncrl.agent_runner import AgentRunner, EpisodeState


class AsyncRunner(AgentRunner):
    def compute_loss(
        self, model: Module, episode_state: EpisodeState
    ) -> Tuple[Tensor, Dict[str, Any]]:
        # obses, dones, actions, net_states, rewards, values, log_probs, entropies = episode_state
        gae = 0.0
        ret = episode_state.values[-1].item()
        R = [0.0] * episode_state.episode_len
        GAE = [0.0] * episode_state.episode_len

        for i in reversed(range(episode_state.episode_len)):
            ret = self.args.gamma * ret + episode_state.rewards[i]
            delta_t = (
                episode_state.rewards[i]
                + self.args.gamma * episode_state.values[i + 1].cpu().item()
                - episode_state.values[i].cpu().item()
            )
            gae = gae * self.args.gamma * self.args.llambda + delta_t

            GAE[i] = gae
            R[i] = ret

        R = torch.from_numpy(np.asarray(R)).float().to(self.args.device)
        GAE = torch.from_numpy(np.asarray(GAE)).float().to(self.args.device)
        log_probs = torch.stack(episode_state.log_probs)
        entropies = torch.stack(episode_state.entropies)
        values = torch.stack(episode_state.values[:-1])

        assert values.shape == R.shape, (
            values.shape,
            R.shape,
        )

        Advs = R - values

        assert Advs.requires_grad
        value_loss = 0.5 * Advs.pow(2).mean()

        assert log_probs.shape == GAE.shape, (
            log_probs.shape,
            GAE.shape,
        )
        assert log_probs.requires_grad

        pg_loss = -(log_probs * GAE.detach()).mean()
        entropy_loss = entropies.mean()

        total_loss = (
            pg_loss
            + self.args.value_loss_coef * value_loss
            - self.args.entropy_coef * entropy_loss
        )
        loss_detail = {
            "entropy_loss": entropy_loss,
            "value_loss": value_loss,
            "pg_loss": pg_loss,
            "total_loss": total_loss,
            "Advs": Advs,
            "GAE": GAE,
        }
        return total_loss, loss_detail

    def log_training(
        self, epoch: int, loss_detail: Dict[str, Tensor], writer: SummaryWriter
    ):
        writer.add_scalar(
            "training/entropy" + str(self.rank),
            loss_detail["entropy_loss"].item(),
            epoch,
        )
        writer.add_scalar(
            "training/value_loss" + str(self.rank),
            loss_detail["value_loss"].item(),
            epoch,
        )
        writer.add_scalar(
            "training/pg_loss" + str(self.rank), loss_detail["pg_loss"].item(), epoch
        )
        writer.add_scalar(
            "training/total_loss" + str(self.rank),
            loss_detail["total_loss"].item(),
            epoch,
        )
        writer.add_scalar(
            "training/grad_norm" + str(self.rank),
            loss_detail["grad_norm"].cpu().item(),
            epoch,
        )
        writer.add_scalar(
            "training/adv" + str(self.rank),
            loss_detail["Advs"].detach().mean().item(),
            epoch,
        )
        writer.add_scalar(
            "training/gae" + str(self.rank),
            loss_detail["GAE"].detach().mean().item(),
            epoch,
        )
