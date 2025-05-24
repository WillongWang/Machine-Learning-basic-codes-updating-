import torch
import gymnasium as gym
from dqn import DQNPixel, DQNTrainer
from wrapper import OnlyPixel, FrameStackTorch

env = gym.make("CartPole-v1", render_mode="rgb_array")
pixel_env = OnlyPixel(env)
pixel_env = gym.wrappers.ResizeObservation(pixel_env, (40, 60)) # 继承gym.ObservationWrapper
pixel_env = FrameStackTorch(pixel_env, 4)

device = "cuda" if torch.cuda.is_available() else "cpu"

obs, _ = pixel_env.reset()
q_net = DQNPixel(obs.shape, 2)
q_net.to(device)

trainer = DQNTrainer(q_net, pixel_env)
trainer.train(2000, batch_size=512)
print("Final test score: ", trainer.eval())
q_net.save("q_net_pixel.pth")