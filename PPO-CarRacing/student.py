import gym
import numpy as np
import json
from collections import deque
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Beta
from neuralArchitecture import NeuralNet
from wrapper import Wrapper
from utils import *
from tqdm import tqdm
img_stack = 4
transition = np.dtype(
    [
        ("state", np.float64, (img_stack, 96, 96)),
        ("action", np.float64, (3,)),
        ("action_logprob", np.float64),
        ("reward", np.float64),
        ("new_state", np.float64, (img_stack, 96, 96)),
    ]
)
class Policy:
    def __init__(self):
        #seed_everything(0)
        self.params = json.load(open("config.json"))
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        print("device: ", self.device)
        self.net = NeuralNet(img_stack).double().to(self.device)
        self.buffer = np.empty(self.params["MAX_SIZE"], dtype=transition)
        self.buffer_counter = 0
        
        self.optimizer = optim.Adam( self.net.parameters(), lr=self.params["LEARNING_RATE"])  
        self.env = gym.make("CarRacing-v2")
        self.state = self.env.reset()
        self.reward_threshold = self.env.spec.reward_threshold
        self.net.load_state_dict(torch.load("./best.pth"))
        self.env_wrap = Wrapper(self.env, self.params["IMG_STACK"])
        self.continuous=True
        self.act_state_stack= None
            
    def save(self, directory, filename, suffix):
        torch.save(
            self.net.state_dict(), "%s/%s_%s.pth" % (directory, filename, suffix)
        )

    def select_action(self, state):
        state = torch.from_numpy(state).double().to(self.device).unsqueeze(0)
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        distribution = Beta(alpha, beta)
        action = distribution.sample()
        log_probs = distribution.log_prob(action).sum(dim=1)

        action = action.squeeze().cpu().numpy()
        log_probs = log_probs.item()
        return action, log_probs

    def update(self):
        D= {
        "state": torch.tensor(self.buffer["state"], dtype=torch.double).to(self.device),
        "action" : torch.tensor(self.buffer["action"], dtype=torch.double).to(self.device),
        "reword" : (
            torch.tensor(self.buffer["reward"], dtype=torch.double)
            .to(self.device)
            .view(-1, 1)
            ),
        "next_s" : torch.tensor(self.buffer["new_state"], dtype=torch.double).to(self.device),
        "old_action_logprobs" : (
            torch.tensor(self.buffer["action_logprob"], dtype=torch.double)
            .to(self.device)
            .view(-1, 1)
            )
        }
        with torch.no_grad():
            target_value = D["reword"] + self.params["GAMMA"] * self.net(D["next_s"])[1]
            advantage = target_value - self.net(D["state"])[1]

        for _ in range(self.params["EPOCH"]):
            for index in BatchSampler(
                SubsetRandomSampler(range(self.params["MAX_SIZE"])),
                self.params["BATCH"],
                False,
            ):
                alpha, beta = self.net(D["state"][index])[0]
                dist = Beta(alpha, beta)
                action_logprobs = dist.log_prob(D["action"][index]).sum(dim=1, keepdim=True)
                probability_ratio = torch.exp(action_logprobs - D["old_action_logprobs"][index])
                uncliped_obj = probability_ratio * advantage[index]
                cliped_obj= (
                    torch.clamp(
                        probability_ratio, 1.0 - self.params["EPS"], 1.0 + self.params["EPS"]
                    )
                    * advantage[index]
                )
                action_loss = -torch.min(
                    uncliped_obj, 
                    cliped_obj
                    ).mean()
                value_loss = F.smooth_l1_loss(self.net(D["state"][index])[1], target_value[index])
                loss = action_loss + 2.0 * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def train(self, n_episodes=5000):
        scores_deque = deque(maxlen=100)
        scores_array = []
        avg_scores_array = []
        state = self.env_wrap.reset()

        progress_bar= tqdm(range(n_episodes))
        for index in progress_bar:
            total_reward = 0
            state = self.env_wrap.reset()

            while True:
                action, log_probs = self.select_action(state)
                next_state, reward, done, _ = self.env_wrap.step(
                    action * np.array([2.0, 1.0, 1.0]) + np.array([-1.0, 0.0, 0.0])
                )

                self.buffer[self.buffer_counter] = state, action, log_probs, reward, next_state
                self.buffer_counter += 1
                if self.buffer_counter >= self.params["MAX_SIZE"]:
                        self.buffer_counter = 0
                        self.update()

                total_reward += reward
                state = next_state
                if done:
                    break

            scores_deque.append(total_reward)
            scores_array.append(total_reward)
            avg_score = np.mean(scores_deque)
            avg_scores_array.append(avg_score)

            progress_bar.set_description(
                f"Score {total_reward}, avg_score {avg_score}")
            if(index % self.params["SAVE_EVERY"]== 0):
                self.save( './', 'model_weights', int(avg_score))
                plot_scores( scores_array, avg_scores_array)
        
        plot_scores( scores_array, avg_scores_array)

    def act(self, state_rgb):
        state_gray = rgb2gray(np.array(state_rgb))
        if(self.act_state_stack is None):
            self.act_state_stack = [state_gray] * img_stack 
                 # four frames for decision
        else:
            self.act_state_stack.pop(0)
            self.act_state_stack.append(state_gray)
        action, _ = self.select_action(np.array(self.act_state_stack))
        action = action * np.array([2.0, 1.0, 1.0]) + np.array([-1.0, 0.0, 0.0])
        return action
