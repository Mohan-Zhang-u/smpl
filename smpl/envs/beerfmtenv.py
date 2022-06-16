# -*- coding: utf-8 -*-
"""
BeerFMT simulates the Beer Fermentation process.
"""

import math
import random

import numpy as np
from gym import spaces, Env
from scipy.integrate import odeint

from .utils import *

MAX_STEPS = 200
random.seed(0)
INIT_SUGAR = 130
BEER_init = [0, 2, 2, INIT_SUGAR, 0, 0, 0, 0]  # X_A, X_L, X_D, S, EtOH, DY, EA = 0, 2, 2, 130, 0, 0, 0
BIOMASS_end_threshold = 0.5
BIOMASS_end_change_threshold = 0.01
SUGAR_end_threshold = 0.5
ZERO_C_IN_K = 273.15



def beer_ode(points, t, sets):
    """
    Beer fermentation process
    """
    X_A, X_L, X_D, S, EtOH, DY, EA = points
    S0, T = sets
    k_x = 0.5 * S0

    u_x0 = math.exp(108.31 - 31934.09 / T)
    Y_EA = math.exp(89.92 - 26589 / T)
    u_s0 = math.exp(-41.92 + 11654.64 / T)
    u_L = math.exp(30.72 - 9501.54 / T)

    u_DY = 0.000127672
    u_AB = 0.00113864

    u_DT = math.exp(130.16 - 38313 / T)
    u_SD0 = math.exp(33.82 - 10033.28 / T)
    u_e0 = math.exp(3.27 - 1267.24 / T)
    k_s = math.exp(-119.63 + 34203.95 / T)

    u_x = u_x0 * S / (k_x + EtOH)
    u_SD = u_SD0 * 0.5 * S0 / (0.5 * S0 + EtOH)
    u_s = u_s0 * S / (k_s + S)
    u_e = u_e0 * S / (k_s + S)
    f = 1 - EtOH / (0.5 * S0)

    dXAt = u_x * X_A - u_DT * X_A + u_L * X_L
    dXLt = -u_L * X_L
    dXDt = -u_SD * X_D + u_DT * X_A
    dSt = -u_s * X_A
    dEtOHt = f * u_e * X_A
    dDYt = u_DY * S * X_A - u_AB * DY * EtOH
    # todo
    dEAt = Y_EA * u_x * X_A
    # dEAt = -Y_EA * u_s * X_A

    return np.array([dXAt, dXLt, dXDt, dSt, dEtOHt, dDYt, dEAt])


class BeerFMTEnvGym(smplEnvBase):
    def __init__(self, dense_reward=True, normalize=True, debug_mode=False, action_dim=1, observation_dim=8,
                 reward_function=None, done_calculator=None, max_observations=[15, 15, 15, 150, 150, 10, 10, MAX_STEPS],
                 min_observations=[0, 0, 0, 0, 0, 0, 0, 0], max_actions=[16.0], min_actions=[9.0],
                 observation_name=["X_A", "X_L", "X_D", "S", "EtOH", "DY", "EA", "time"], action_name=["temperature"], np_dtype=np.float32, max_steps=MAX_STEPS, 
                 error_reward=-float(MAX_STEPS)):

        """
        Time is in our observation_space. We make the env time aware.
        The only action/input is temperature.
        The observations are Active Biomass (g/L), Lag Biomass (g/L), Dead Biomass (g/L), Sugar (g/L), Ethanol (g/L), Diacetyl (g/L), Ethyl Acetate (g/L), time (Hours)
        """
        
        # define arguments
        self.step_count = 0
        self.total_reward = 0
        self.done = False
        self.dense_reward = dense_reward
        self.normalize = normalize  
        self.debug_mode = debug_mode  
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.reward_function = reward_function  
        self.done_calculator = done_calculator  
        self.max_observations = max_observations
        self.min_observations = min_observations
        self.max_actions = max_actions
        self.min_actions = min_actions
        self.observation_name = observation_name
        self.action_name = action_name
        if self.observation_name is None:
            self.observation_name = [f'o_{i}' for i in range(self.observation_dim)]
        if self.action_name is None:
            self.action_name = [f'a_{i}' for i in range(self.action_dim)]
        self.np_dtype = np_dtype
        self.max_steps = max_steps
        self.error_reward = error_reward
        if self.reward_function is None:
            self.reward_function = self.reward_function_standard
        if self.done_calculator is None:
            self.done_calculator = self.done_calculator_standard

        # define the state and action spaces
        self.max_observations = np.array(self.max_observations, dtype=self.np_dtype)
        self.min_observations = np.array(self.min_observations, dtype=self.np_dtype)
        self.max_actions = np.array(self.max_actions, dtype=self.np_dtype)
        self.min_actions = np.array(self.min_actions, dtype=self.np_dtype)
        if self.normalize:
            self.observation_space = spaces.Box(low=-1, high=1, shape=(self.observation_dim,))
            self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,))
        else:
            self.observation_space = spaces.Box(low=self.min_observations, high=self.max_observations,
                                                shape=(self.observation_dim,))
            self.action_space = spaces.Box(low=self.min_actions, high=self.max_actions, shape=(self.action_dim,))
        
        self.res_forplot = []  # for plotting purposes
    
    def reward_function_standard(self, previous_observation, action, current_observation, reward=None):
        if reward is not None:
            return reward
        elif self.observation_beyond_box(current_observation) or self.action_beyond_box(action):
            return self.error_reward

        done, done_info = self.done_calculator(current_observation, self.step_count, reward)
        early_submission = done_info.get('early_submission', False)
        if early_submission:
            reward = -self.error_reward # early submission is rewarded. The end goal in this simulation is to reach the stop condition (finish production) with a certain time limit, the quicker the better.
        elif done:  
            reward = self.error_reward # reaches time limit but reaction has not finished
        else:
            reward = -1 # should finish as soon as possible
        
        reward = max(self.error_reward, reward)  # reward cannot be smaller than the error_reward
        if self.debug_mode:
            print("reward:", reward)
        return reward
    
    def done_calculator_standard(self, current_observation, step_count, reward, update_prev_biomass=False, done=None, done_info=None):
        if done is None:
            done = False
        else:
            if done_info is not None:
                return done, done_info
            else:
                raise Exception("When done is given, done_info should also be given.")

        if done_info is None:
            done_info = {"terminal": False, "timeout": False}
        else:
            if done_info["terminal"] or done_info["timeout"]:
                done = True
                return done, done_info

        if self.observation_beyond_box(current_observation):
            done_info["terminal"] = True
            done = True
        if reward == self.error_reward:
            done_info["terminal"] = True
            done = True
        if step_count >= self.max_steps:  # same as range(0, max_steps)
            done_info["terminal"] = True
            done_info["timeout"] = True
            done = True
            
        X_A, X_L, X_D, S, EtOH, DY, EA, time = current_observation
        current_biomass = X_A + X_L + X_D
        if current_biomass < BIOMASS_end_threshold or abs(current_biomass - self.prev_biomass) < BIOMASS_end_change_threshold:
            if S < SUGAR_end_threshold:
                if EtOH > 50.0:
                    done_info["terminal"] = True # this is still terminal, though should be rewarded.
                    done_info["early_submission"] = True
                    done = True
        if update_prev_biomass:
            self.prev_biomass = current_biomass
        return done, done_info
    
    def sample_initial_state(self): # 
        init_X_L = np.random.uniform(2 * 0.9, 2 * 1.1) # around 2
        init_X_D = np.random.uniform(2 * 0.9, 2 * 1.1) # around 2
        init_SUGER = np.random.uniform(INIT_SUGAR * 0.9, INIT_SUGAR * 1.1) # around INIT_SUGAR
        observation = np.array([0, init_X_L, init_X_D, init_SUGER, 0, 0, 0, 0], dtype=self.np_dtype)
        return observation
    
    def reset(self, initial_state=None, normalize=None):
        """
        required by gym.
        This function resets the environment and returns an initial observation.
        """
        self.step_count = 0
        self.total_reward = 0
        self.done = False

        if initial_state is not None:
            initial_state = np.array(initial_state, dtype=self.np_dtype)
            observation = initial_state
            self.init_observation = initial_state
        else:
            observation = self.sample_initial_state()
            self.init_observation = observation
        self.previous_observation = observation
        
        # TOMODIFY: reset your environment here.
        self.prev_biomass = observation[0] + observation[1] + observation[2]
        
        normalize = self.normalize if normalize is None else normalize
        if normalize:
            observation, _, _ = normalize_spaces(observation, self.max_observations, self.min_observations)
        return observation
    
    def step(self, action, normalize=None):
        """
        required by gym.
        This function performs one step within the environment and returns the observation, the reward, whether the episode is finished and debug information, if any.
        """
        if self.debug_mode:
            print("action:", action)
        reward = None
        done = None
        done_info = {"terminal": False, "timeout": False}
        action = np.array(action, dtype=self.np_dtype)
        normalize = self.normalize if normalize is None else normalize
        if normalize:
            action, _, _ = denormalize_spaces(action, self.max_actions, self.min_actions)

        # TOMODIFY: proceed your environment here and collect the observation.
        t = np.arange(0 + self.step_count, 1 + self.step_count, BIOMASS_end_change_threshold)
        X_A, X_L, X_D, S, EtOH, DY, EA, _ = self.previous_observation
        sol = odeint(beer_ode, (X_A, X_L, X_D, S, EtOH, DY, EA), t, args=([INIT_SUGAR, action[0] + ZERO_C_IN_K],))
        self.res_forplot.append(sol[-1, :])
        X_A, X_L, X_D, S, EtOH, DY, EA = sol[-1, :]
        observation = [X_A, X_L, X_D, S, EtOH, DY, EA, self.step_count + 1]

        observation = np.array(observation, dtype=self.np_dtype)
        if not reward:
            reward = self.reward_function(self.previous_observation, action, observation, reward=reward)
        if not done:
            done, done_info = self.done_calculator(observation, self.step_count, reward, update_prev_biomass=True, done=done, done_info=done_info)
        self.previous_observation = observation

        self.total_reward += reward
        if self.dense_reward:
            reward = reward  # conventional
        elif not done:
            reward = 0.0
        else:
            reward = self.total_reward
        # clip observation so that it won't be beyond the box
        observation = observation.clip(self.min_observations, self.max_observations)
        if normalize:
            observation, _, _ = normalize_spaces(observation, self.max_observations, self.min_observations)
        self.step_count += 1
        info = {"res_forplot": np.array(self.res_forplot)}
        info.update(done_info)
        return observation, reward, done, info

