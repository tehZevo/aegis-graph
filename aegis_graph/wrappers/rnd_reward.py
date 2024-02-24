import gymnasium as gym

class RNDReward(gym.Wrapper):
    def __init__(self, env, rnd):
        super().__init__(env)
        self.rnd = rnd
    
    def step(self, action):
        obs, reward, done, terminated, info = self.env.step(action)
        rnd_reward = self.rnd.step(obs)
        #TODO: RND scale?
        return obs, reward + rnd_reward, done, terminated, info