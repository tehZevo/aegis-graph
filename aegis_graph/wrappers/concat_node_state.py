import gymnasium as gym
import numpy as np

class ConcatNodeState(gym.ObservationWrapper):
    def __init__(self, env, node):
        super().__init__(env)
        self.node = node

        wrapped_obs_size = self.env.observation_space.shape[-1]
        node_size = self.node.get_state().shape[0]
        new_obs_size = wrapped_obs_size + node_size

        #TODO: customizable range?
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=[new_obs_size])
    
    def observation(self, obs):
        return np.concatenate([obs, self.node.state], -1)