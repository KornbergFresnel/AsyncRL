# AsyncRL

A repository for the reproduction of async RL algorithms. Each application under the directory [/applications/](/applications) is built for solving a spefic domain/task. And scripts under [/examples/](/examples) give cases for training.

## Reason for making this repository

These days, I want to investigate the use of [FeuDal](https://arxiv.org/abs/1703.01161), a typical hierarchical RL algorithm for goal-conditioned learning. However, there is no official implementation for this work, and many third-party implementations totally cannot work for solving Atari games. I list them below to help users to get rid of wasting their time reading this garbage:

- [lweitkamp/feudalnets-pytorch](https://github.com/lweitkamp/feudalnets-pytorch)
- [davidhershey/feudal_networks](https://github.com/davidhershey/feudal_networks)
- [dnddnjs/feudal-montezuma](https://github.com/dnddnjs/feudal-montezuma)
- [vtalpaert/pytorch-feudal-network](https://github.com/vtalpaert/pytorch-feudal-network)

After a deeper review of their code, I found the reason is that there is a misleading of the pytorch-implementation of [A3C](https://arxiv.org/abs/1602.01783), an asynchronous RL algorithm as the basis of FeuDal. And I also list them as follows (totally cannot work for solving Gym, too!):

- [ikostrikov/pytorch-a3c](https://github.com/ikostrikov/pytorch-a3c)
- [MorvanZhou/pytorch-A3C](https://github.com/MorvanZhou/pytorch-A3C)

**NOTE**: never implement A3C with a shared optimizer like the above! Please refer to the official torch guides of [IMPLEMENTING BATCH RPC PROCESSING USING ASYNCHRONOUS EXECUTIONS](https://pytorch.org/tutorials/intermediate/rpc_async_execution.html)

In this repo, I give a functional and lightweight implementation of A3C, which works for solving gym, at least!

## Algorithm Table

Status | Algorithm | Example 
--- | --- | ---
DONE | A3C | PYTHONPATH=.:asyncrl python examples/a3c_gym.py --num-processes 2
🕓 | [IMPALA](https://arxiv.org/abs/1802.01561) | ...
🕓 | [APPO (Sample-Factory implementation)](http://proceedings.mlr.press/v119/petrenko20a/petrenko20a.pdf) | ...

## Evaluation Results

**Solving CartPole-v1 via A3C (`num_worker=2`)**

![](/images/a3c_cart_pole_num_worker_2.png)
