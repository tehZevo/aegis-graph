import os

from .node import Node
from .link import Link

class Graph:
    def __init__(self, save_every=1000, save_path="graph_data"):
        self.nodes = []
        self.save_every = save_every
        self.save_steps = 0
        self.save_path = save_path
    
    def create_node(self, size):
        node = Node(size)
        self.nodes.append(node)
        return node
    
    def add_node(self, node):
        self.nodes.append(node)
    
    def remove_node(self, node):
        if node not in self.nodes:
            raise ValueError("Node not in graph")

        self.nodes.remove(node)
    
    def update(self):
        for node in self.nodes:
            node.update()
        
        self.save_steps += 1
        if self.save_steps >= self.save_every:
            self.save(self.save_path)
            self.save_steps = 0
    
    def get_source(self, id):
        #TODO: how to load non-node sources?
        #TODO: this assumes all sources are nodes
        for node in self.nodes:
            if node.id == id:
                return node

        raise ValueError(f"No source with id '{id}' in graph")
    
    def get_node(self, id):
        for node in self.nodes:
            if node.id == id:
                return node

        raise ValueError(f"No node with id '{id}' in graph")


    def save(self, path):
        for node in self.nodes:
            node.save(os.path.join(path, "nodes"))

    def load_links(self, node_path):
        #TODO: link loading is probably overly complicated
        #TODO: create link using node.link, then just load in models and id
        for fn in os.listdir(os.path.join(node_path, "links")):
            print(f"Loading link '{fn}'...")
            link_path = f = os.path.join(node_path, "links", fn)
            link = Link.load(link_path, self)
            link.node.add_link(link)

    def load(self, path):
        #load nodes
        for fn in os.listdir(os.path.join(path, "nodes")):
            print(f"Loading node '{fn}'...")
            node_path = os.path.join(path, "nodes", fn)
            node = Node.load(node_path)
            self.add_node(node)

        #load links
        for fn in os.listdir(os.path.join(path, "nodes")):
            print(f"Loading node '{fn}' links...")
            node_path = os.path.join(path, "nodes", fn)
            self.load_links(node_path)
        