import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import time

from aegis_graph import Node

node = Node(32)

node.link(node)

while True:
    time.sleep(1)
    node.update()
    print(node.get_state())
    print("tock")