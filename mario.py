import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import time

import gymnasium as gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import cv2
import numpy as np
from scipy.special import softmax

from aegis_graph import Graph, ManualSensor

ACTION_REPEAT = 8
EXPERIMENT_NAME = "single_no_recur"

#downscale by 8x8 (30, 32) and scale to 0..1 range
class PixelMarioWrapper(gym.Wrapper):
    def __init__(self, env, scale_size=32):
        self.env = env
        self.scale_size = scale_size
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=[scale_size, scale_size, 3])
        # self.action_space = gym.spaces.Box(low=-1, high=1, shape=[4])
    
    def resize(self, obs):
        return cv2.resize(obs, (self.scale_size, self.scale_size), interpolation=cv2.INTER_AREA) / 255.
        
    #manually specifying step because gym 0.24.1 compat
    def step(self, action):
        #32, 30 because (w, h) format of cv2
        obs, reward, done, info = self.env.step(action)
        obs = self.resize(obs)

        return obs, reward, done, False, info
    
    def reset(self):
        obs = self.env.reset()
        obs = self.resize(obs)
        return obs, {}

mario = gym_super_mario_bros.make("SuperMarioBros-v0")
mario = JoypadSpace(mario, COMPLEX_MOVEMENT)
mario = PixelMarioWrapper(mario, scale_size=32)

print(mario.observation_space)
print(mario.action_space)

mario_sensor = ManualSensor(np.prod(mario.observation_space.shape))

graph = Graph(save_every=1000, save_path=EXPERIMENT_NAME)

# center_node = graph.create_node(128)
# center_node.link(mario_sensor)

# aux_node = graph.create_node(1024)
# aux_node.link(center_node)
# center_node.link(aux_node)

mario_action = graph.create_node(mario.action_space.n)
# mario_action = graph.create_node(mario.action_space.shape[0])
# mario_action.link(center_node)
mario_action.link(mario_sensor, recurrent=False, agent_type="DDPG")

#TODO: async update?

obs, _ = mario.reset()

cv2.namedWindow(EXPERIMENT_NAME, cv2.WINDOW_NORMAL)

def transform_action(action, method="softmax"):
    #since all node outputs are continuous, create a pseudo softmax weighted random action choice
    if method == "softmax":
        action = action * 5
        action = softmax(action)
        action = np.random.choice(len(action), p=action)
    else:
        action = np.argmax(action)
    return action

ACTION_NAMES = ["NOOP", "RIGHT", "RIGHT A", "RIGHT B", "RIGHT A B", "A", "LEFT", "LEFT A", "LEFT B", "LEFT A B", "DOWN", "UP"]
def transform_action_new(action):
    right = action[0] > 0.5
    left = action[0] < -0.5
    up = action[1] > 0.5
    down = action[1] < -0.5
    a = action[2] > 0
    b = action[3] > 0

    if up: return 11
    if down: return 10
    if right and a and b: return 4
    if right and a: return 2
    if right and b: return 3
    if right: return 1
    if left and a and b: return 9
    if left and a: return 7
    if left and b: return 8
    if left: return 6
    if a: return 5
    return 0

steps = 0
while True:
    #update mario sensor
    obs = obs.flatten()
    mario_sensor.set_state(obs)

    graph.update()

    steps += 1
    
    action = mario_action.get_state()
    action = transform_action(action, method="softmax")
    print(steps, ACTION_NAMES[action])

    for _ in range(ACTION_REPEAT):
        obs, reward, done, terminated, info = mario.step(action)
        if done or terminated:
            obs, _ = mario.reset()
            break
    
    img = obs.astype("float32")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow(EXPERIMENT_NAME, img)
    cv2.waitKey(1)
    