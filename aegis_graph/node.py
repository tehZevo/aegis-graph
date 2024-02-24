from uuid import uuid4
import os

import numpy as np

from .source import Source
from .link import Link

#TODO: how to link node to self without redundant inputs
class Node(Source):
    def __init__(self, size, save_every=1000, save_path="nodes"):
        super().__init__()
        #TODO: other initializers?
        self.state = np.zeros([size])
        self.links = {}
        self.id = str(uuid4())
        self.save_every = save_every
        self.save_steps = 0
        self.save_path = save_path
    
    def save(self, path):
        #TODO: save config
        for link in self.links.values():
            link.save(os.path.join(path, self.id))

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

        self.save_steps += 1
        if self.save_steps >= self.save_every:
            self.save(self.save_path)
            self.save_steps = 0

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
        