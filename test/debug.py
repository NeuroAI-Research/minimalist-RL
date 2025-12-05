import gymnasium as gym
import numpy as np
import torch.nn as nn
from spinup.sac import MLPActorCritic, ReplayBuffer
from trading_models.utils import shape, tensor

from minimalist_RL.SAC import ActorCritic
from minimalist_RL.utils import RLData, set_seed


def test1():
    env = gym.make("HalfCheetah-v5")
    n1 = MLPActorCritic(env.observation_space, env.action_space)
    n2 = ActorCritic(env, Act=nn.ReLU)
    n2.load_state_dict(n1.state_dict())
    obs = tensor(env.reset()[0]).reshape((1, -1))
    for n in [n1, n2]:
        n: ActorCritic
        set_seed()
        act, logp = n.pi(obs)
        q1, q2 = n.q1(obs, act), n.q2(obs, act)
        name = n.__class__.__name__
        print(f"act.sum: {act.sum()}, logp: {logp}, q1: {q1}, q2: {q2}. {name}")


def test2():
    obs_dim, act_dim, cap = (3, 4), (5, 6), int(1e6)
    b1 = ReplayBuffer(obs_dim, act_dim, cap)
    b2 = RLData({}, cap)
    rand = np.random.rand
    for _ in range(1000):
        data = {
            "obs": rand(*obs_dim),
            "act": rand(*act_dim),
            "rew": rand(),
            "obs2": rand(*obs_dim),
            "done": rand(),
        }
        b1.store(*data.values())
        b2.push(data)
    set_seed()
    d1 = b1.sample_batch(100)
    set_seed()
    d2 = b2.sample(100)
    print({k: v.sum() for k, v in d1.items()})
    print({k: v.sum() for k, v in d2._data.items()})
    print(shape(d1))
    print(d2)


if __name__ == "__main__":
    test1()
    test2()
