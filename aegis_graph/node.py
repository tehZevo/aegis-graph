import os

import numpy as np
import yaml

from .source import Source
from .link import Link

class Node(Source):
    def load(path):
        with open(os.path.join(path, "config.yml"), "r") as f:
            config = yaml.safe_load(f.read())
        
        node = Node(config["size"])
        node.id = config["id"]

        return node
        
    def __init__(self, size):
        super().__init__()
        #TODO: other initializers?
        self.state = np.zeros([size])
        self.links = {}
    
    def save(self, path):
        node_path = os.path.join(path, self.id)
        os.makedirs(node_path, exist_ok=True)
        
        #save config
        config = {
            "size": self.state.shape[0],
            "id": self.id
        }
        with open(os.path.join(node_path, "config.yml"), "w") as f:
            f.write(yaml.dump(config))
        
        #save state
        np.save(os.path.join(node_path, "state"), self.state)

        #save links
        for link in self.links.values():
            link.save(os.path.join(node_path, "links"))

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

        #TODO: if save issues occur, split the continue_step.set()s into a graph-triggered function

        #step envs
        for link in self.links.values():
            link.env.continue_step.set()
            
    def link(self, source, recurrent=True):
        #TODO: allow this?
        if source.id in self.links:
            raise ValueError("That source is already in this node's links")

        link = Link(source, self, recurrent=recurrent)
        self.links[source.id] = link
        link.start()
        
        return link
    
    def add_link(self, link):
        #TODO: ensure source is not linked already
        self.links[link.source.id] = link
        link.start()
    
    def unlink(self, source):
        if source.id not in self.links:
            raise ValueError("That source is not present in this node's links")
            
        del self.links[source.id]
    
    def get_state(self):
        return self.state
        