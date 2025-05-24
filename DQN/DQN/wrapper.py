import gymnasium as gym
import numpy as np

class OnlyPixel(gym.ObservationWrapper): # 看ObservationWrapper源码,可如env一样ObservationWrapper.reset()/step(); property Wrapper.action_space: Return the Wrapper.env(__init__中的).action_space unless overwritten then the wrapper action_space is used. (property Wrapper.observation_space同理)
    """                                  # Wrapper继承Env
    Wrapping PixelObservationWrapper to make it really pixel only.
    """
    def __init__(self, env):
        super().__init__(env)
        self.env = gym.wrappers.AddRenderObservation(env, render_only=True) # 有gym.wrappers.AddRenderObservation(env, render_only=False).observation_space["pixels"]
    
    def observation(self, observation):
        return observation # observation – An element of the environment’s observation_space

class FrameStackTorch(gym.ObservationWrapper):
    """
    Wrapping FrameStack to produce torch compatible input
    """
    def __init__(self, env, n_frames):
        super().__init__(env)
        self.env = gym.wrappers.FrameStackObservation(env, n_frames)
        obs_shape = self.observation_space.shape
        new_shape = (obs_shape[0] * obs_shape[-1], *obs_shape[1:-1])
        self.observation_space = gym.spaces.Box(
            low = self.observation_space.low.astype(np.float32).reshape(*new_shape) / 255, # Box源码有self.low/high
            high = self.observation_space.high.astype(np.float32).reshape(*new_shape) / 255,
            dtype = np.float32,
            shape = new_shape
        )

    def observation(self, observation):
        arr = np.array(observation, dtype=np.float32) / 255
        arr = np.moveaxis(arr, -1, 0)
        return arr.reshape(*self.observation_space.shape)

