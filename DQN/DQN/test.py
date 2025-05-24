import gymnasium as gym
import torch
from dqn import DQN, DQNTrainer, EpsGreedy
from itertools import count

device = "cuda" if torch.cuda.is_available() else "cpu"

env = gym.make("CartPole-v1", render_mode="human")

dim_state = env.observation_space.shape[0]
n_actions = env.action_space.n

q_net = DQN(dim_state, n_actions)
q_net.load("q_net.pth")
q_net.to(device)

policy = EpsGreedy(q_net)
total_reward = 0
state, _ = env.reset()
for _ in count():
    action = policy.get_action(state)
    state, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break
print("Total reward: ", total_reward)
env.close()
