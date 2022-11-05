from typing import Dict

import torchvision.transforms.functional as TF
import torch

from utils import dotdict


class AbstractAgent:
    default_options = dotdict(
        max_episode_length=4000,
    )

    def __init__(self, env, options: Dict = None):
        # self.eval_stats = []
        # self.train_stats = []
        self.cur_stats = dotdict()
        self.env = env
        self.options = self.default_options.copy()
        if options is not None:
            self.options.update(options)

    def _to_torch(self, img):
        return TF.to_tensor(img)

    def step(self, state):
        obs, reward, done, _, _ = self.env.step(self.act(state))
        self.cur_stats.reward += reward
        self.cur_stats.steps += 1
        return obs, reward, done, None, None

    def reset(self, train=True, options: Dict = None):
        # reset environment
        obs, _ = self.env.reset(options=options)

        # reset current statistics
        self.cur_stats = dotdict(
            reward=0,
            steps=0,
        )
        # add current statistics to training or evaluation
        # if train:
        #     self.train_stats.append(self.cur_stats)
        # else:
        #     self.eval_stats.append(self.cur_stats)
        return obs, None

    def eval(self):
        obs, _ = self.reset()
        for _ in range(self.options.max_episode_length):
            obs, reward, done, _, _ = self.step(obs)
            self.env.render()

            if done:
                break

        self.env.close()

    # ABSTRACT METHODS

    def act(self, state):
        pass

    def train(self):
        pass


# class DecisionTransformer(torch.nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self,x):
#         pass

#     def run(self,x):
#         y = self(x)
#         return y
