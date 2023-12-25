from typing import Type
from argparse import Namespace

import threading

import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
import torchvision

from torch import optim


num_classes, batch_update_size = 30, 5


class ParameterServer:
    def __init__(
        self,
        args,
        model_cls,
        model_kwargs,
        batch_update_size: int = batch_update_size,
        batch_mode: str = "avg",
    ):
        self.model: nn.Module = model_cls(**model_kwargs)
        self.model.train()
        self.lock = threading.Lock()
        self.future_model = torch.futures.Future()
        # NOTE the batch update size would be better for the same as worker number
        self.batch_update_size = batch_update_size
        self.curr_update_size = 0
        self.batch_mode = batch_mode

        if args.optimizer == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=args.lr, momentum=0.9
            )
        elif args.optimizer == "rmsprop":
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=args.lr)
        else:
            raise NotImplementedError

        self.reset_grad()

    def get_model(self) -> nn.Module:
        return self.model

    def reset_grad(self):
        for p in self.model.parameters():
            p.grad = torch.zeros_like(p)

    @staticmethod
    @rpc.functions.async_execution
    def update_and_fetch_model(ps_rref, worker_id, grads):
        # Using the RRef to retrieve the local PS instance
        self = ps_rref.local_value()

        with self.lock:
            self.curr_update_size += 1
            # accumulate gradients into .grad field
            for p, g in zip(self.model.parameters(), grads):
                assert g is not None
                p.grad += g

            # Save the current future_model and return it to make sure the
            # returned Future object holds the correct model even if another
            # thread modifies future_model before this thread returns.
            fut = self.future_model

            if self.curr_update_size >= self.batch_update_size:
                # update the model
                if self.batch_mode == "avg":
                    for p in self.model.parameters():
                        p.grad /= self.batch_update_size
                self.curr_update_size = 0
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.reset_grad()
                # by settiing the result on the Future object, all previous
                # requests expecting this updated model will be notified and
                # the their responses will be sent accordingly.
                fut.set_result(self.model)
                self.future_model = torch.futures.Future()

        return fut


param_server = None
global_lock = threading.Lock()


def get_parameter_server(
    args: Namespace,
    model_class: Type,
    model_kwargs: dict,
    worker_num: int,
    batch_mode: str = "avg",
):
    global param_server
    with global_lock:
        if not param_server:
            param_server = ParameterServer(
                args,
                model_class,
                model_kwargs,
                batch_update_size=worker_num,
                batch_mode=batch_mode,
            )
        return param_server


def run_parameter_server(rank: int, world_size: int, ps_name: int):
    # The parameter server just acts as a host for the model and responds to
    # requests from trainers, hence it does not need to run a loop.
    # rpc.shutdown() will wait for all workers to complete by default, which
    # in this case means that the parameter server will wait for all trainers
    # to complete, and then exit.
    print("PS master initializing RPC")
    rpc.init_rpc(name=ps_name, rank=rank, world_size=world_size)
    print("RPC initialized! Running parameter server...")
    rpc.shutdown()
    print("RPC shutdown on parameter server.")
