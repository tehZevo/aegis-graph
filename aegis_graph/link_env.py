from threading import Event

import gymnasium as gym

class LinkEnv(gym.Env):
    def __init__(self, source, node):
        super().__init__()
        self.source = source
        self.node = node
        self.last_action = None
        self.action_updated = Event()
        self.continue_step = Event()

        #TODO: customizable range for observation?
        #TODO: some way to specify source size without get_state first?
        source_size = source.get_state().shape[0]
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=[source_size])
        
        node_size = node.get_state().shape[0]
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=[node_size])
    
    def step(self, action):
        self.last_action = action

        #tell node we're ready
        self.action_updated.set()

        #wait here for the node to unlock us
        self.continue_step.wait()
        self.continue_step.clear()
        
        obs = self.source.get_state()

        return obs, 0, False, False, {}
    
    def reset(self, **kwargs):
        obs = self.source.get_state()
        return obs, {}