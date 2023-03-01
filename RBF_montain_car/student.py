import random
import numpy as np
import gym
import time
from gym import spaces
import os
import pickle
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler

class VanillaFeatureEncoder:
    def __init__(self, env):
        self.env = env
        
    def encode(self, state):
        return state
    
    @property
    def size(self): 
        return self.env.observation_space.shape[0]

class RBFFeatureEncoder:
    def __init__(self, env): # modify
        self.env = env
        n_components=1000
        gamma_step= 0.5
        n_encoders=10
       
        samplers_list=[]
        for i in range (1,n_encoders):
            samplers_list.append( ("rbf"+str(i), RBFSampler(gamma=gamma_step* i, n_components=n_components) )  )

        observed_samples = np.array([env.observation_space.sample() for x in range(10000)])
        scaler = StandardScaler()
        scaler.fit(observed_samples)
                
        RBF_featurizer = FeatureUnion(samplers_list)
        feature_examples = RBF_featurizer.fit_transform(scaler.transform(observed_samples))

        self.dimensions = feature_examples.shape[1]
        self.scaler = scaler
        self.featurizer = RBF_featurizer
        
    def encode(self, state): # modify
       
        scaled = self.scaler.transform([state])
        transformed= self.featurizer.transform(scaled)
        return transformed[0]
    
    @property
    def size(self): # modify
        # return the correct size of the observation
        return  self.dimensions

class TDLambda_LVFA:
    def __init__(self, env, feature_encoder_cls=RBFFeatureEncoder,
        alpha=0.005, alpha_decay=1, 
        gamma=0.9999, epsilon=0.3, epsilon_decay=0.99, lambda_=0.9):
        self.env = env
        self.feature_encoder = feature_encoder_cls(env)
        self.shape = (self.env.action_space.n, self.feature_encoder.size)
        self.weights = np.random.random(self.shape)
        self.traces = np.zeros(self.shape)
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.lambda_ = lambda_
        
    def Q(self, feats):
        feats = feats.reshape(-1,1)
        return self.weights@feats
    
    def update_transition(self, s, action, s_prime, reward, done): # modify
        s_feats = self.feature_encoder.encode(s)
        s_prime_feats = self.feature_encoder.encode(s_prime)
        action_prime = self.epsilon_greedy(s_prime)
        td_error = reward


        if not done:
            td_error += self.gamma*self.Q(s_prime_feats)[action_prime]
        td_error -= self.Q(s_feats)[action]
        self.traces[action]*= self.lambda_ * self.gamma
        self.traces[action]+= s_feats 
        self.weights[action] += self.alpha *(td_error * self.traces[action])


    def update_alpha_epsilon(self): # modify
        self.epsilon =  self.epsilon*self.epsilon_decay
        self.alpha = self.alpha*self.alpha_decay
        

    def epsilon_greedy(self, state, epsilon=None):  # modify
        if epsilon is None: epsilon = self.epsilon
        if random.random()<epsilon:
            return self.env.action_space.sample()
        return self.policy(state)
        
    def policy(self, state): # do not touch
        state_feats = self.feature_encoder.encode(state)
        return self.Q(state_feats).argmax()
           
    def train(self, n_episodes=200, max_steps_per_episode=200): # do not touch
        for episode in range(n_episodes):
            done = False
            s, _ = self.env.reset()
            self.traces = np.zeros(self.shape)
            for i in range(max_steps_per_episode):            
                action = self.epsilon_greedy(s)
                s_prime, reward, done, _, _ = self.env.step(action)
                self.update_transition(s, action, s_prime, reward, done)          
                s = s_prime 
                if done: break
                
            self.update_alpha_epsilon()

            if episode % 20 == 0:
                print(episode, self.evaluate(), self.epsilon, self.alpha)
                
    def evaluate(self, env=None, n_episodes=10, max_steps_per_episode=200): # do not touch
        if env is None:
            env = self.env
            
        rewards = []
        for episode in range(n_episodes):
            total_reward = 0
            done = False
            s, _ = env.reset()
            for i in range(max_steps_per_episode):
                action = self.policy(s)             
                s_prime, reward, done, _, _ = env.step(action)
                total_reward += reward
                s = s_prime
                if done: break
            
            rewards.append(total_reward)
            
        return np.mean(rewards)

    def save(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, fname):
        return pickle.load(open(fname,'rb'))

