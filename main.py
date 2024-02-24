import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import time

from aegis_graph import Node

a = Node(32)

b = Node(128)

c = Node(16)

b.link(a)
c.link(a)

nodes = [a, b, c]

#TODO: async update

while True:
    time.sleep(1)
    for node in nodes:
        node.update()
    print(b.get_state())
    print("tock")