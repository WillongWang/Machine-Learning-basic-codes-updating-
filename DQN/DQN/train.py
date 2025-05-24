import gymnasium as gym
import torch
from dqn import DQN, DQNTrainer

device = "cuda" if torch.cuda.is_available() else "cpu"

env = gym.make("CartPole-v1", render_mode="rgb_array")
dim_state = env.observation_space.shape[0]
n_actions = env.action_space.n
q_net = DQN(dim_state, n_actions)
q_net.to(device)
trainer = DQNTrainer(q_net, env)

trainer.train(2400)
print("Final test score: ", trainer.eval())
q_net.save("q_net.pth")
