from typing import Dict, List

import torchvision.transforms.functional as TF
import torch

from environments.pytux import PyTux
from utils import dotdict
import utils
import pathlib
import numpy as np


class AbstractAgent:
    default_options = dotdict(
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

    def step(self, action):
        '''The function takes in an action, takes a step in the environment, and returns the observation, reward, done, and info.
        
        The observation is the state of the environment. 
        
        The reward is the reward for the action taken. 
        
        The done is a boolean that indicates whether the episode is over. 
        
        The info is a dictionary that contains auxiliary diagnostic information. 
        
        Parameters
        ----------
        action
            The action to take in the environment.
        
        Returns
        -------
            obs, reward, done, None, None
        
        '''
        obs, reward, done, _, _ = self.env.step(action)
        self.cur_stats.cum_reward += reward
        self.cur_stats.steps += 1
        return obs, reward, done, None, None

    def reset(self, options: Dict = None):
        '''The function resets the environment and the current statistics
        
        Parameters
        ----------
        options : Dict
            dictionary of options for the environment
        
        Returns
        -------
            The initial state
        
        '''
        if options is not None:
            self.options.update(options)

        # reset environment
        obs, _ = self.env.reset(options=self.options)

        # reset current statistics
        self.cur_stats = dotdict(
            cum_reward=0,
            steps=0,
        )
        return obs, None

    def evaluate(self):
        self.eval_mode(True)

        obs, _ = self.reset()
        while(True):
            action = self.act(obs)
            obs, reward, done, _, _ = self.step(action)

            if done:
                break
        
        return self.cur_stats

    def save_run(self, path:str, n_runs=1, noise=None, save_every_steps:int = 10):
        '''It runs the agent for a number of episodes, and saves the data at regular intervals
        
        Parameters
        ----------
        path : str
            the path where the data will be saved
        n_runs, optional
            number of runs to save
        noise : optional
            noise value to add
        save_every_steps : int, optional
            how often to save the data
        
        '''
        for run in range(n_runs):
            print(f"Run {run} on {self}")
            # constant value summed to the timestep 
            # to randomize when image taken for each run
            save_rand = int(np.random.rand() * save_every_steps)

            # path where the run will be saved
            p = f"{path}/{run}"
            pathlib.Path(p).mkdir(parents=True, exist_ok=True)

            obs, _ = self.reset()
            t=0
            while(True):
                action = self.act(obs, noise=noise)
                next_obs, reward, done, _, _ = self.step(action)

                # save data
                if (t+1) % save_every_steps == save_rand:
                    utils.save_dict(dict(
                        state=obs,
                        action=action,
                        reward=reward,
                        cum_reward=self.cur_stats.cum_reward,
                    ), path=f"{p}/{t}.pt")

                obs = next_obs
                t+=1
                if done:
                    break

            # save final reward and results
            self.cur_stats.done = done
            # utils.save_dict(self.cur_stats, path=f"{p}/final.pt")
            utils.save_dict(self.cur_stats, path=f"{p}/final.txt", as_str=True)


    # ABSTRACT METHODS

    def act(self, state, noise=None):
        '''The function takes in a state and an optional noise parameter, and returns the action taken by the agent
        
        Parameters
        ----------
        state
            The current state of the game.
        noise : optional
            The amount of noise to add.
        
        Returns
        -------
            PyTux.Action()
        
        '''
        return PyTux.Action()

    def train(self):
        pass
    
    def eval_mode(self,eval:bool = True):
        pass
