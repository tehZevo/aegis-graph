from threading import Thread
import traceback
from uuid import uuid4
import os

import gymnasium as gym
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from rnd import RND
import yaml

from .link_env import LinkEnv
from .wrappers import ConcatNodeState, RNDReward

class Link:
    def load(link_path, graph):
        with open(os.path.join(link_path, "config.yml"), "r") as f:
            config = yaml.safe_load(f.read())

        source = graph.get_source(config["source_id"])
        node = graph.get_node(config["node_id"])
        agent_type = config["agent_type"]
        
        link = Link(
            source,
            node,
            recurrent=config["recurrent"],
            agent_type=agent_type
        )

        #TODO: have to provide env
        if agent_type == "PPO":
            link.agent = PPO.load(os.path.join(link_path, "agent"))
        elif agent_type == "DDPG":
            link.agent = DDPG.load(os.path.join(link_path, "agent"))
        else:
            raise ValueError(f"Unknown agent_type '{agent_type}'")
            
        link.rnd = RND()
        link.rnd.load(os.path.join(link_path, "rnd"))

        return link

    def __init__(self, source, node, recurrent=True, agent_type="PPO"):
        self.source = source
        self.node = node
        self.recurrent = recurrent
        self.agent_type = agent_type
        self.id = str(uuid4())
        self.env = None
        self.agent = None
        self.rnd = None

    def save(self, path):
        link_path = os.path.join(path, self.id)
        os.makedirs(link_path, exist_ok=True)

        #save config
        config = {
            "source_id": self.source.id,
            "node_id": self.node.id,
            "recurrent": self.recurrent,
            "agent_type": self.agent_type
        }
        with open(os.path.join(link_path, "config.yml"), "w") as f:
            f.write(yaml.dump(config))

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
                if self.recurrent:
                    wrapped_env = ConcatNodeState(wrapped_env, self.node)
                #TODO: customizable ppo parameters
                if self.agent_type == "PPO":
                    vec_env = DummyVecEnv([lambda: wrapped_env])
                    self.agent = PPO("MlpPolicy", vec_env, verbose=0, n_steps=128, n_epochs=1)
                elif self.agent_type == "DDPG":
                    self.agent = DDPG("MlpPolicy", wrapped_env, verbose=0)
                else:
                    raise ValueError(f"Unknown agent_type '{self.agent_type}'")
                self.agent.learn(total_timesteps=float("inf"))
            except Exception as e:
                print(traceback.format_exc())

        #TODO: store thread in links dict?
        self.thread = Thread(target=run, daemon=True)
        self.thread.start()