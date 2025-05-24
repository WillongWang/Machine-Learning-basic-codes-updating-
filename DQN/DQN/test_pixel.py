import gymnasium as gym
import torch
from dqn import DQNPixel, DQNTrainer, EpsGreedy
from wrapper import OnlyPixel, FrameStackTorch
from itertools import count

device = "cuda" if torch.cuda.is_available() else "cpu"

env = gym.make("CartPole-v1", render_mode="rgb_array")
pixel_env = OnlyPixel(env)
pixel_env = gym.wrappers.ResizeObservation(pixel_env, (40, 60))
pixel_env = FrameStackTorch(pixel_env, 4)

in_shape = pixel_env.observation_space.shape
n_actions = pixel_env.action_space.n

q_net = DQNPixel(in_shape, n_actions)
q_net.load("q_net_pixel.pth")
q_net.to(device)

policy = EpsGreedy(q_net)
total_reward = 0
state, _ = pixel_env.reset()
for _ in count():
    action = policy.get_action(state)
    state, reward, terminated, truncated, info = pixel_env.step(action)
    total_reward += reward
    if terminated or truncated:
        break
print("Total reward: ", total_reward)
env.close()
