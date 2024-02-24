from threading import Thread
import traceback

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from rnd import RND

from .source import Source
from .link_env import LinkEnv
from .wrappers import ConcatNodeState, RNDReward

class Node(Source):
    def __init__(self, size):
        super().__init__()
        #TODO: other initializers?
        self.state = np.zeros([size])
        self.links = {}
    
    def update(self):
        #dont update state if we have no links
        if len(self.links) == 0:
            print("nothing to update...")
            return
            
        actions = [env.last_action for env in self.links.values()]
        #TODO: other methods of reduce?
        self.state = np.mean(actions, axis=0)

        #step envs
        for link_env in self.links.values():
            link_env.lock.release()
            
    def link(self, source):
        #TODO: allow this?
        if source.id in self.links:
            raise ValueError("That source is already in this node's links")

        link_env = LinkEnv(source, self)
        self.links[source.id] = link_env
        self.start_link(link_env)
    
    def unlink(self, source):
        if source.id not in self.links:
            raise ValueError("That source is not present in this node's links")
            
        del self.links[source.id]
    
    def get_state(self):
        return self.state
    
    def start_link(self, link_env):
        def run():
            try:
                #TODO: store RND somewhere so it can be saved?
                source_size = link_env.source.get_state().shape[-1]
                rnd = RND(source_size)
                wrapped_env = link_env
                wrapped_env = RNDReward(wrapped_env, rnd)
                wrapped_env = ConcatNodeState(wrapped_env, self)
                vec_env = DummyVecEnv([lambda: wrapped_env])
                #TODO: save callback
                model = PPO("MlpPolicy", vec_env, verbose=1)
                model.learn(total_timesteps=float("inf"))
            except Exception as e:
                print(traceback.format_exc())

        
        #TODO: store thread in links dict?
        Thread(target=run, daemon=True).start()



