import gymnasium as gym

class RNDReward(gym.Wrapper):
    def __init__(self, env, rnd):
        super().__init__(env)
        self.rnd = rnd
    
    def step(self, action):
        obs, reward, done, terminated, info = self.env.step(action)
        #NOTE: reward should be 0 since link env doesnt have a reward
        rnd_reward = self.rnd.step(obs)
        return obs, reward + rnd_reward, done, terminated, info