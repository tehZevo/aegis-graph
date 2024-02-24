import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import time

import gymnasium as gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import cv2
import numpy as np

from aegis_graph import Graph, ManualSensor

ACTION_REPEAT = 8
EXPERIMENT_NAME = "double_agent"

#downscale by 8x8 (30, 32) and scale to 0..1 range
class PixelMarioWrapper(gym.Wrapper):
    def __init__(self, env):
        self.env = env
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=[30, 32, 3])
    
    #manually specifying step because gym 0.24.1 compat
    def step(self, action):
        #32, 30 because (w, h) format of cv2
        obs, reward, done, info = self.env.step(action)
        obs = cv2.resize(obs, (32, 30)) / 255.

        return obs, reward, done, False, info
    
    def reset(self):
        obs = self.env.reset()
        obs = cv2.resize(obs, (32, 30)) / 255.
        return obs, {}

mario = gym_super_mario_bros.make("SuperMarioBros-v3")
mario = JoypadSpace(mario, COMPLEX_MOVEMENT)
mario = PixelMarioWrapper(mario)

print(mario.observation_space)
print(mario.action_space)

mario_sensor = ManualSensor(np.prod(mario.observation_space.shape))

graph = Graph(save_every=1000, save_path=EXPERIMENT_NAME)

a = graph.create_node(1024)
a.link(mario_sensor)

b = graph.create_node(mario.action_space.n)
b.link(a)
# b.link(mario_sensor)

#TODO: async update?

obs, _ = mario.reset()

cv2.namedWindow(EXPERIMENT_NAME, cv2.WINDOW_NORMAL)

steps = 0
while True:
    #update mario sensor
    obs = obs.flatten()
    mario_sensor.set_state(obs)

    graph.update()

    steps += 1
    print(steps, np.argmax(b.get_state()))
    
    action = b.get_state()
    action = np.argmax(action)

    for _ in range(ACTION_REPEAT):
        obs, reward, done, terminated, info = mario.step(action)
        if done or terminated:
            obs, _ = mario.reset()
            break
    
    img = obs.astype("float32")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow(EXPERIMENT_NAME, img)
    cv2.waitKey(1)
    