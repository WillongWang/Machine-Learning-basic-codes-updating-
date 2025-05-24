import torch
import torch.nn as nn
import numpy as np
from collections import deque, namedtuple
import random, copy
from itertools import count


class DQNBase(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.n_actions = n_actions
        self.net = None

    def q_vals(self, states, actions): # states形状(batch_size,dim_state), actions形状(batch_size,1)
        # Q(s,a)
        return self.forward(states).gather(1, actions.reshape(-1, 1).to(torch.int64)).reshape(-1)

    def v_vals(self, states): # states形状(batch_size,dim_state)
        # max_a Q(s,a)
        return self.forward(states).amax(-1)

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

    @property
    def device(self):
        return next(self.net.parameters()).device

    def get_compat(self, x):
        x = np.stack(x)
        x = torch.tensor(x, device=self.device)
        return x

class DQN(DQNBase):
    def __init__(self, dim_state, n_actions):
        super().__init__(n_actions)
        self.dim_state = dim_state
        self.net = nn.Sequential(
            nn.Linear(dim_state, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, n_actions),
        )
        self.forward = self.net.forward

class DQNPixel(DQNBase):
    def __init__(self, in_shape, n_actions):
        super().__init__(n_actions)
        self.in_shape = in_shape
        self.conv = nn.Sequential(
            nn.Conv2d(in_shape[0], 64, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(-3) # (Batch, C, H, W)变成(C, H, W)吧
        )
        conv_out_shape = self.conv(torch.rand((1, *in_shape))).shape
        self.fc = nn.Sequential(
            nn.Linear(conv_out_shape[-1], self.n_actions),
        )
        self.net = nn.Sequential(self.conv, self.fc)
        self.forward = self.net.forward


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayBuffer:
    def __init__(self, capacity):
        self.deque = deque(maxlen=capacity)

    def push(self, *arg):
        self.deque.append(Transition(*arg))

    def sample(self, batch_size):
        return random.sample(self.deque, batch_size)

    def __len__(self):
        return len(self.deque)
        
    def count_final(self):
        finals = [*filter(lambda trans: trans.next_state is None, self.deque)]
        return len(finals)

class EpsGreedy:
    def __init__(self, q_net):
        self.q_value_net = q_net # DQNBase

    def get_action(self, state, mode="eval", eps=0.1): # state大小为dim_state
        if mode == "train" and random.random() < eps:
            return random.randint(0, self.q_value_net.n_actions - 1)
        else:
            state = self.q_value_net.get_compat(state)
            return self.q_value_net(state).argmax().item() # torch.argmax(input, dim, keepdim=False) → LongTensor; dim (int) – the dimension to reduce. If None, the argmax of the flattened input is returned.


class DQNTrainer:
    def __init__(
        self,
        q_net, # DQNBase
        env,
        lr=1e-4,
        gamma=0.99,
        replay_buf_size=10000,
        eps_start=0.5,
        eps_end=0.01,
        eps_decay=1e4,
        update_thres=1000,
    ):

        self.q_net = q_net
        self.env = env
        self.opt = torch.optim.RMSprop(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.criterion = torch.nn.SmoothL1Loss()
        self.policy = EpsGreedy(q_net)
        self.replay_buf = ReplayBuffer(replay_buf_size)
        self.target_net = copy.deepcopy(self.q_net)
        self.counter = 0

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.update_thres = update_thres

    @property
    def eps(self):
        ratio = np.exp(-self.counter / self.eps_decay)
        eps = self.eps_end + (self.eps_start - self.eps_end) * ratio
        return eps

    def update(self, batch_size):
        if batch_size > len(self.replay_buf):
            return
        batch = self.replay_buf.sample(batch_size) # replay_buf见下train()
        # check next_state to tell if is final
        non_final_mask = [*map(lambda trans: trans.next_state is not None, batch)]
        state, action, next_state, reward = zip(*batch)
        next_state = [*filter(lambda x: x is not None, next_state)]

        batch = Transition(
            *map(self.q_net.get_compat, [state, action, next_state, reward])
        )
        # Q^(s,a)
        state_action_vals = self.q_net.q_vals(batch.state, batch.action)
        # V(s')
        next_state_vals = torch.zeros_like(state_action_vals)
        with torch.no_grad():
            next_state_vals[non_final_mask] = self.target_net.v_vals(batch.next_state)
        # Q(s,a)=r + gamma*V(s')
        expected_state_action_vals = batch.reward + self.gamma * next_state_vals

        # compute loss & update
        loss = self.criterion(state_action_vals, expected_state_action_vals)
        self.opt.zero_grad()
        loss.backward()
        for para in self.q_net.parameters():
            para.grad.data.clamp_(-1, 1) # Tensor.grad, Tensor.__init__(self, data)
        self.opt.step()

    def train(self, n_episodes, batch_size=512):
        for i in range(n_episodes):
            if self.replay_buf.count_final() < len(self.replay_buf)/batch_size:
                print("Few final state contained in replay buffer, training done!")
                break
            state, _ = self.env.reset()
            total_reward = 0
            for _ in count():
                self.counter += 1
                action = self.policy.get_action(state, "train", self.eps)
                next_state, reward, terminated, truncated, info = self.env.step(action)

                total_reward += reward
                if terminated:
                    next_state = None

                self.replay_buf.push(state, action, next_state, reward)

                state = next_state
                self.update(batch_size)
                if self.counter % self.update_thres == 0:
                    self.target_net.load_state_dict(
                        copy.deepcopy(self.q_net.state_dict())
                    )

                if terminated or truncated:
                    break

            if i % 200 == 0:
                print(
                    "Episode: {:6}, Total_reward: {:7}, eps: {:6.4}, Total_reward_eval: {:7}".format(
                        i, total_reward, self.eps, self.eval()
                    )
                )

    def eval(self, render=True):
        state, _ = self.env.reset()
        total_reward = 0
        for _ in count():
            action = self.policy.get_action(state)
            state, reward, terminated, truncated, info = self.env.step(action)
            if render:
                img = self.env.render()

            total_reward += reward
            if terminated or truncated:
                break
        return total_reward
