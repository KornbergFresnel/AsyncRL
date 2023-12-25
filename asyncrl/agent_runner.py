from typing import Any, Tuple, Dict
from itertools import count
from argparse import Namespace
from collections import namedtuple

import time
import threading

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from torch.distributions import Categorical

from tensorboardX import SummaryWriter

from asyncrl.parameter_server import ParameterServer


EpisodeState = namedtuple(
    "EpisodeState",
    "obses, dones, actions, net_states, rewards, values, log_probs, entropies, episode_len",
)


class AgentRunner:
    def __init__(
        self,
        args: Namespace,
        ps_rref: Any,
        rank: int,
        model_class: nn.Module,
        model_kwargs: dict,
        make_env: Any,
        log_dir: str,
    ) -> None:
        self.args = args
        self.ps_rref = ps_rref
        self.rank = rank
        self.worker_name = rpc.get_worker_info().name
        self.model = model_class(**model_kwargs).to(args.device)
        self.device = args.device
        self.env = make_env()
        self.log_dir = log_dir

    def run_episode(
        self,
        obs: np.ndarray,
        global_counter: mp.Value = None,
        lock: threading.Lock = None,
    ) -> EpisodeState:
        done = False

        rewards = []
        values = []
        log_probs = []
        entropies = []
        obses = []
        actions = []
        dones = []
        net_states = []

        # make sure it is not None
        net_state = self.model.init_state(1, self.device)

        counter = count() if not self.model.training else range(self.args.num_steps)

        for step_cnt in counter:
            obses.append(obs)
            net_states.append([e.squeeze(0) for e in net_state])
            obs = torch.from_numpy(obs).float().to(self.device)
            value, logits, net_state = self.model(obs.unsqueeze(0), net_state)

            dist = Categorical(logits=logits)
            if self.model.training:
                action = dist.sample()
            else:
                action = logits.argmax(dim=-1)

            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            obs, reward, done, truncated, info = self.env.step(action.cpu().numpy()[0])
            done = done or truncated

            values.append(value.squeeze())
            actions.append(action.cpu().numpy())
            log_probs.append(log_prob.squeeze())
            entropies.append(entropy.squeeze())
            rewards.append(reward)
            dones.append(done)

            if global_counter:
                with lock:
                    global_counter.value += 1

            if done:
                break

        obses.append(obs)
        net_states.append([e.squeeze(0) for e in net_state])

        if dones[-1]:
            values.append(torch.zeros(1).to(self.device).squeeze())
        else:
            obs = torch.from_numpy(obs).float().to(self.device)
            value, _, _ = self.model(obs.unsqueeze(0), net_state)
            values.append(value.squeeze())

        return EpisodeState(
            obses,
            dones,
            actions,
            net_states,
            rewards,
            values,
            log_probs,
            entropies,
            len(rewards),
        )

    def update_and_fetch_model(self, model: nn.Module) -> nn.Module:
        for p in model.parameters():
            if p.grad is None:
                raise RuntimeError(
                    f"Empty grad at worker: {self.worker_name} {self.rank}"
                )

        model: nn.Module = rpc.rpc_sync(
            self.ps_rref.owner(),
            ParameterServer.update_and_fetch_model,
            args=(
                self.ps_rref,
                self.worker_name,
                [p.grad for p in model.cpu().parameters()],
            ),
        ).to(self.device)
        return model

    def fetch_model(self) -> nn.Module:
        return self.ps_rref.rpc_sync().get_model().to(self.device)

    def test(self, counter: mp.Value, lock: threading.Lock):
        start_time = time.time()
        writer = SummaryWriter(log_dir=self.log_dir)

        for epoch in count():
            self.model: nn.Module = self.fetch_model()
            self.model.eval()
            # always reset
            obs, _ = self.env.reset()
            # test should not update counter
            episode_state = self.run_episode(obs)
            reward_sum = sum(episode_state.rewards)
            print(
                "Time {}, eval epoch {}, training steps {}, FPS {:.0f}, episode reward {}, episode length {}".format(
                    time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)),
                    epoch,
                    counter.value,
                    counter.value / (time.time() - start_time),
                    reward_sum,
                    episode_state.episode_len,
                )
            )
            with lock:
                for name, param in self.model.named_parameters():
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
                writer.add_scalar(
                    "evaluation/episode_reward", reward_sum, counter.value
                )
            time.sleep(5)

    def compute_loss(
        self, model: nn.Module, episode_state: EpisodeState
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        raise NotImplementedError

    def log_training(
        self, epoch: int, loss_detail: Dict[str, torch.Tensor], writer: SummaryWriter
    ):
        pass

    def train(self, counter: mp.Value = None, lock: threading.Lock = None):
        self.model: nn.Module = self.fetch_model()
        time.sleep(1)
        writer = SummaryWriter(log_dir=self.log_dir)

        obs, _ = self.env.reset()

        for epoch in count():
            self.model.train()
            episode_state = self.run_episode(obs, counter, lock)

            with lock:
                writer.add_scalars(
                    "training/episode_info" + str(self.rank),
                    {
                        "episode_reward": sum(episode_state.rewards),
                        "episode_length": episode_state.episode_len,
                    },
                    epoch,
                )

            total_loss, loss_detail = self.compute_loss(self.model, episode_state)

            assert total_loss.requires_grad

            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.args.max_grad_norm
            )

            loss_detail["grad_norm"] = grad_norm

            # then sync model
            self.model = self.update_and_fetch_model(self.model)
            assert self.model.training

            with lock:
                self.log_training(epoch, loss_detail, writer)

            if episode_state.dones[-1]:
                obs, _ = self.env.reset()
            else:
                obs = episode_state.obses[-1]
