import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import time

import numpy as np

from aegis_graph import Graph

graph = Graph(save_every=10, save_path="load_test")

a = graph.create_node(32)
b = graph.create_node(32)
c = graph.create_node(32)

a.link(b)
a.link(c)
b.link(c)
b.link(a)
c.link(a)
c.link(b)

print("running graph")
for _ in range(10):
    graph.update()

del graph

print("loading graph")
graph = Graph()
graph.load("load_test")

print(len(graph.nodes))
print(len(graph.nodes[0].links))