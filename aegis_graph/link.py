from threading import Thread
import traceback
from uuid import uuid4
import os

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rnd import RND

from .link_env import LinkEnv
from .wrappers import ConcatNodeState, RNDReward

class Link:
    def __init__(self, source, node):
        self.source = source
        self.node = node
        self.env = None
        self.agent = None
        self.rnd = None
        self.id = str(uuid4())

    def save(self, path):
        self.agent.save(os.path.join(path, self.id, "agent"))
        self.rnd.save(os.path.join(path, self.id, "rnd"))
    
    def start(self):
        def run():
            try:
                self.env = LinkEnv(self.source, self.node)
                source_size = self.source.get_state().shape[-1]
                self.rnd = RND(source_size)
                wrapped_env = self.env
                wrapped_env = RNDReward(wrapped_env, self.rnd)
                wrapped_env = ConcatNodeState(wrapped_env, self.node)
                vec_env = DummyVecEnv([lambda: wrapped_env])
                #TODO: save callback (save RND in callback too)
                self.agent = PPO("MlpPolicy", vec_env, verbose=1, n_steps=128, n_epochs=4)
                self.agent.learn(total_timesteps=float("inf"))
            except Exception as e:
                print(traceback.format_exc())

        #TODO: store thread in links dict?
        self.thread = Thread(target=run, daemon=True)
        self.thread.start()