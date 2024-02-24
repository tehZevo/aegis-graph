import numpy as np

from .source import Source
from .link import Link

#TODO: how to link node to self without redundant inputs
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
        
        #wait for all of our links
        for link in self.links.values():
            link.env.action_updated.wait()
            link.env.action_updated.clear()

        actions = [link.env.last_action for link in self.links.values()]
        #TODO: other methods of reduce?
        self.state = np.mean(actions, axis=0)

        #step envs
        for link in self.links.values():
            link.env.continue_step.set()
            
    def link(self, source):
        #TODO: allow this?
        if source.id in self.links:
            raise ValueError("That source is already in this node's links")

        link = Link(source, self)
        self.links[source.id] = link
        link.start()
    
    def unlink(self, source):
        if source.id not in self.links:
            raise ValueError("That source is not present in this node's links")
            
        del self.links[source.id]
    
    def get_state(self):
        return self.state
    
    # def start_link(self, link_env):
    #     def run():
    #         try:
    #             #TODO: store RND somewhere so it can be saved?
    #             source_size = link_env.source.get_state().shape[-1]
    #             rnd = RND(source_size)
    #             wrapped_env = link_env
    #             wrapped_env = RNDReward(wrapped_env, rnd)
    #             wrapped_env = ConcatNodeState(wrapped_env, self)
    #             vec_env = DummyVecEnv([lambda: wrapped_env])
    #             #TODO: save callback
    #             model = PPO("MlpPolicy", vec_env, verbose=1, n_steps=128, n_epochs=4)
    #             model.learn(total_timesteps=float("inf"))
    #         except Exception as e:
    #             print(traceback.format_exc())

    #     #TODO: store thread in links dict?
    #     Thread(target=run, daemon=True).start()
