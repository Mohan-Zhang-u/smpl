import codecs
import csv
import math
import random
import sys

import numpy as np
from gym import spaces, Env
from mzutils import get_things_in_loc
from pensimpy.peni_env_setup import PenSimEnv

from .utils import *

csv.field_size_limit(sys.maxsize)
MINUTES_PER_HOUR = 60
BATCH_LENGTH_IN_MINUTES = 230 * MINUTES_PER_HOUR
BATCH_LENGTH_IN_HOURS = 230
STEP_IN_MINUTES = 12
STEP_IN_HOURS = STEP_IN_MINUTES / MINUTES_PER_HOUR
NUM_STEPS = int(BATCH_LENGTH_IN_MINUTES / STEP_IN_MINUTES)
WAVENUMBER_LENGTH = 2200


def get_observation_data_reformed(observation, t):
    """
    Get observation data at t.
    vars are Temperature,Acid flow rate,Base flow rate,Cooling water,Heating water,Vessel Weight,Dissolved oxygen concentration 
    respectively in csv terms, but used abbreviation here to stay consistent with peni_env_setup
    """
    vars = ['T', 'Fa', 'Fb', 'Fc', 'Fh', 'Wt', 'DO2']
    pH = observation.pH.y[t]
    pH = -math.log(pH) / math.log(10) if pH != 0 else pH
    return [t * STEP_IN_MINUTES / MINUTES_PER_HOUR, pH] + [
        eval(f"observation.{var}.y[t]", {'observation': observation, 't': t}) for var in vars]


class PenSimEnvGym(PenSimEnv, smplEnvBase):
    def __init__(
            self, recipe_combo, dense_reward=True, normalize=True, debug_mode=False, action_dim=6, observation_dim=9,
            reward_function=None, done_calculator=None,
            max_observations=[552.0, 16.10523, 725.6828, 13.717274, 540.0, 3600.0002, 1892.07874, 253840.11, 47.898834],
            min_observations=[0.0, 0.0, 118.98977, 0.0, 0.0, 0.0, 0.0, 25003.258, 0.0],
            max_actions=[4100.0, 151.0, 36.0, 76.0, 1.2, 510.0],
            min_actions=[0.0, 7.0, 21.0, 29.0, 0.5, 0.0],
            observation_name=None, action_name=None, initial_state_deviation_ratio = 0.1,
            np_dtype=np.float32, max_steps=NUM_STEPS, error_reward=-100.0,
            fast=True, random_seed=0, random_seed_max=20000):
        """
        Time is not in our observation_space. We make the env time unaware and MDP.
        _max_episode_steps and random_seed_ref are for PenSimEnv usage.
        random_seed_max is the max random seed to be used in the sample_initial_state.
        
        Args:
            recipe_combo (RecipeCombo): The recipe combo defined for the PenSimEnv.
        
        Attributes:
            random_seed_ref (int): The random seed used in the sample_initial_state. 
        """
        super(PenSimEnvGym, self).__init__(recipe_combo, fast=fast)
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
            self.action_name = ["Discharge rate", ",Sugar feed rate", "Soil bean feed rate", "Aeration rate", "Back pressure", "Water injection/dilution"] # discharge, Fs, Foil, Fg, pressure, Fw
        self.initial_state_deviation_ratio = initial_state_deviation_ratio
        self.np_dtype = np_dtype
        self.max_steps = max_steps
        self._max_episode_steps = max_steps
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
        random.seed(random_seed)
        self.random_seed_ref = random_seed
        self.random_seed_max = random_seed_max
        self.init_observation_from_dataset = np.array([0.2, 6.5, 300.0, 0.0, 10.828493, 0.0001, 150.0, 62500.0, 14.75], dtype=self.np_dtype) # init observation generated from dataset.
        
    def sample_initial_state(self, random_seed_ref=None):
        # notice that this function here has to reset the PenSimEnv.
        if random_seed_ref:
            self.random_seed_ref = random_seed_ref
        else:
            self.random_seed_ref = random.randint(3, self.random_seed_max)
        _, x = super().reset()
        self.x = x
        observation = get_observation_data_reformed(x, 0)
        observation = np.array(observation, dtype=np.float32)
        return observation

    def reset(self, normalize=None, random_seed_ref=None):
        self.step_count = 0
        self.total_reward = 0
        self.done = False
        
        observation_0s = self.sample_initial_state(random_seed_ref=random_seed_ref) # this observation is all zeros.
        observation = self.init_observation_from_dataset
        self.init_observation = observation
        self.previous_observation = observation
        
        normalize = self.normalize if normalize is None else normalize
        if normalize:
            observation, _, _ = normalize_spaces(observation, self.max_observations, self.min_observations)
        return observation

    def step(self, action, normalize=None):
        
        if self.debug_mode:
            print("action:", action)
        reward = None
        done_info = None
        action = np.array(action, dtype=self.np_dtype)
        normalize = self.normalize if normalize is None else normalize
        if normalize:
            action, _, _ = denormalize_spaces(action, self.max_actions, self.min_actions)
        
        try:
            self.step_count += 1 # here we increment at front
            values_dict = self.recipe_combo.get_values_dict_at(self.step_count * STEP_IN_MINUTES)
            # served as a batch buffer below
            pensimpy_observation, x, yield_per_run, done = super().step(self.step_count, self.x, action[1], action[2], action[3],
                                                                        action[4], action[0], action[5],
                                                                        values_dict['Fpaa'])
            # in pensimpy, done = True if k == NUM_STEPS else False
            if done:
                done_info = {"timeout": True, "error_occurred": False, "terminal": True}
            else:
                done_info = {"timeout": False, "error_occurred": False, "terminal": False}
            reward = yield_per_run
            self.x = x
            observation = get_observation_data_reformed(x, self.step_count - 1)
        except Exception as e:
            observation = self.min_observations
            done_info = {"timeout": False, "error_occurred": True, "terminal": True}
        
        observation, reward, done, done_info = self.observation_done_and_reward_calculator(observation, action, normalize=normalize, step_reward=reward, done_info=done_info)
        self.step_count -= 1 # we already increment at front.
        info = {}
        info.update(done_info)
        return observation, reward, done, info


class PeniControlData:
    """
    dataset class helper, mainly aims to mimic d4rl's qlearning_dataset format (which returns a dictionary).
    produced from PenSimPy generated csvs.
    """

    def __init__(self, load_just_a_file='', dataset_folder='examples/example_batches', delimiter=',', observation_dim=9,
                 action_dim=6, normalize=True, np_dtype=np.float32) -> None:
        """
        :param dataset_folder: where all dataset csv files are living in
        """
        self.dataset_folder = dataset_folder
        self.delimiter = delimiter
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.normalize = normalize
        self.np_dtype = np_dtype
        self.max_observations = [276.0, 8.052615, 362.8414, 6.858637, 270.0, 1800.0001, 946.03937, 126920.055, 23.949417]
        self.min_observations = [0.16000001, 4.5955915, 237.97954, 0.0, 0.0, 0.0, 0.0, 50006.516, 2.3598127]
        self.max_actions = [4100.0, 151.0, 36.0, 76.0, 1.2, 510.0]
        self.min_actions = [0.0, 7.0, 21.0, 29.0, 0.5, 0.0]
        self.max_observations = np.array(self.max_observations, dtype=self.np_dtype)
        self.min_observations = np.array(self.min_observations, dtype=self.np_dtype)
        self.max_actions = np.array(self.max_actions, dtype=self.np_dtype)
        self.min_actions = np.array(self.min_actions, dtype=self.np_dtype)
        
        if load_just_a_file != '':
            file_list = [load_just_a_file]
        else:
            file_list = get_things_in_loc(dataset_folder, just_files=True)
        self.file_list = file_list

    def load_file_list_to_dict(self, file_list, shuffle=True):
        file_list = file_list.copy()
        random.shuffle(file_list)
        dataset = {}
        observations = []
        actions = []
        next_observations = []
        rewards = []
        terminals = []
        for file_path in file_list:
            tmp_observations = []
            tmp_actions = []
            tmp_next_observations = []
            tmp_rewards = []
            tmp_terminals = []
            with codecs.open(file_path, 'r', encoding='utf-8') as fp:
                csv_reader = csv.reader(fp, delimiter=self.delimiter)
                next(csv_reader)
                # get rid of the first line containing only titles
                for row in csv_reader:
                    observation = [row[0]] + row[7:-1]
                    # there are 9 items: Time Step, pH,Temperature,Acid flow rate,Base flow rate,Cooling water,Heating water,Vessel Weight,Dissolved oxygen concentration
                    assert len(observation) == self.observation_dim
                    action = [row[1], row[2], row[3], row[4], row[5], row[6]]
                    # there are 6 items: Discharge rate,Sugar feed rate,Soil bean feed rate,Aeration rate,Back pressure,Water injection/dilution
                    assert len(action) == self.action_dim
                    reward = row[-1]
                    terminal = False
                    tmp_observations.append(observation)
                    tmp_actions.append(action)
                    tmp_rewards.append(reward)
                    tmp_terminals.append(terminal)
            tmp_terminals[-1] = True
            tmp_next_observations = tmp_observations[1:] + [tmp_observations[-1]]
            observations += tmp_observations
            actions += tmp_actions
            next_observations += tmp_next_observations
            rewards += tmp_rewards
            terminals += tmp_terminals
        dataset['observations'] = np.array(observations, dtype=np.float32)
        dataset['actions'] = np.array(actions, dtype=np.float32)
        dataset['next_observations'] = np.array(next_observations, dtype=np.float32)
        dataset['rewards'] = np.array(rewards, dtype=np.float32)
        dataset['terminals'] = np.array(terminals, dtype=bool)
        self.dataset_max_observations = dataset['observations'].max(axis=0)
        self.dataset_min_observations = dataset['observations'].min(axis=0)
        self.dataset_max_actions = dataset['actions'].max(axis=0)
        self.dataset_min_actions = dataset['actions'].min(axis=0)
        print("max observations:", self.max_observations)
        print("min observations:", self.min_observations)
        print("dataset max observations:", self.dataset_max_observations)
        print("dataset min observations:", self.dataset_min_observations)
        print("max actions:", self.max_actions)
        print("min actions:", self.min_actions)
        print("dataset max actions:", self.dataset_max_actions)
        print("dataset min actions:", self.dataset_min_actions)
        print("normalize:", self.normalize)
        print("using max/min observations and actions.")
        if self.normalize:
            dataset['observations'], _, _ = normalize_spaces(dataset['observations'], self.max_observations,
                                                            self.min_observations)
            dataset['next_observations'], _, _ = normalize_spaces(dataset['next_observations'], self.max_observations,
                                                                self.min_observations)
            dataset['actions'], _, _ = normalize_spaces(dataset['actions'], self.max_actions,
                                                        self.min_actions)  # passed in a normalized version.
        # self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,))
        return dataset

    def get_dataset(self):
        return self.load_file_list_to_dict(self.file_list)
