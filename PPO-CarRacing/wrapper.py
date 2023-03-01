from utils   import rgb2gray
import numpy as np

class Wrapper():

    def __init__(self, env, img_stack=4):
        self.env = env  
        self.img_stack=img_stack
        self.action_repeat = 10

    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()
        img_rgb,_ = self.env.reset()
        img_gray = rgb2gray(np.array(img_rgb))
        self.stack = [img_gray] * self.img_stack  # four frames for decision
        return np.array(self.stack)

    def step(self, action):
        total_reward = 0
        for _ in range(self.action_repeat):
            img_rgb, reward, die, _, _ = self.env.step(action)
            # don't penalize "die state"
            if die:
                reward += 100
            # green penalty
            if np.mean(img_rgb[:, :, 1]) > 185.0:
                reward -= 0.05
            total_reward += reward
            # if no reward recently, end the episode
            done = True if self.av_r(reward) <= -0.1 else False
            if done or die:
                break
        img_gray = rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)
        assert len(self.stack) == self.img_stack
        return np.array(self.stack), total_reward, done, die
        
    @staticmethod
    def reward_memory():
        # record reward for last 100 steps
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)
        return memory