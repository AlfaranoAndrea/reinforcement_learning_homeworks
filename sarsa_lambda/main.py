import numpy as np
import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
from statistics import mean

from student import sarsa_lambda
import random


def evaluate(map_env, num_episodes):

    env = gym.make("FrozenLake-v1", desc=map_env, render_mode="ansi")
    env_render = gym.make("FrozenLake-v1", desc=map_env, render_mode="ansi")

    Q = sarsa_lambda(env)
    rewards = []
    for ep in range(num_episodes):
        tot_reward = 0
        done = False
        s, _ = env_render.reset()
        while not done:
            a = np.argmax(Q[s])
            s, r, done, _, _ = env_render.step(a)
            tot_reward += r
        #print("\tTotal Reward ep {}: {}".format(ep, tot_reward))
        rewards.append(tot_reward)
    return mean(rewards)



if __name__ == '__main__':
    result= []
    map_env = ["SFFFHF", "FFFFFF", "FHFFFH", "FFFFFF", "HFHFFG"]
    for i in range(50):
        print("________________________")
        print(f"START EPISODE N= {i}")
        num_episodes = 10

        

        mean_rew =evaluate(map_env, num_episodes)
        result.append(mean_rew)
        print("Mean reward over {} episodes: {}".format(num_episodes, mean_rew))
    mean=0.0
    for i in result:
        mean+=i
    mean= mean/len(result)
    print(f"mean of all episode={mean}")