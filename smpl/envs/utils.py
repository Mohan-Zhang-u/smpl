from torch.utils.data import Dataset
import json
import os
import random
import math
import time
# ---- to capture numpy warnings ----
import warnings

import matplotlib.pyplot as plt
import numpy as np
from gym import spaces, Env  # to create an openai-gym environment https://gym.openai.com/
from mzutils import SimplePriorityQueue, normalize_spaces, denormalize_spaces, mkdir_p
from scipy.integrate import solve_ivp  # the ode solver
from tqdm import tqdm

class TorchDatasetFromD4RL(Dataset):
    def __init__(self, dataset_d4rl) -> None:
        import d3rlpy
        """
        dataset_d4rl should be a dictionary in the d4rl dataset format.
        """
        self.dataset = d3rlpy.dataset.MDPDataset(dataset_d4rl['observations'], dataset_d4rl['actions'],
                                                 dataset_d4rl['rewards'], dataset_d4rl['terminals'])

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        episode = self.dataset.__getitem__(idx)
        return {'observations': episode.observations, 'actions': episode.actions, 'rewards': episode.rewards}


class smplEnvBase(Env):

    def __init__(
            self, dense_reward=True, normalize=True, debug_mode=False, action_dim=2, observation_dim=3,
            max_observations=[1.0, 1.0],
            min_observations=[-1.0, -1.0],
            max_actions=[1.0, 1.0],
            min_actions=[-1.0, -1.0],
            observation_name=None, action_name=None, initial_state_deviation_ratio=None, np_dtype=np.float32,
            max_steps=None, error_reward=-100.0):
        """the __init__ of a smpl environment.

        Args:
            dense_reward (bool, optional): Whether returns a dense reward or not. If True, will try to return a reward for each step. If False, will return a reward at the end of the episode. Defaults to True.
            normalize (bool, optional): Whether to normalize the actions taken and observations returned to a certain range. If True, the range $$\in R^n, [-1, 1]^n$$ where n is the dimension of an action/observation. Defaults to True.
            debug_mode (bool, optional): Whether to return or print extra information. Defaults to False.
            action_dim (int, optional): TOMODIFY: Dimensionality of an action. Defaults to 2.
            observation_dim (int, optional): TOMODIFY: Dimensionality of an observation. Defaults to 3.
            reward_function ([type], optional): Provide to replace with your custom reward function. When not given, use environments' default reward function. Defaults to None.
            done_calculator ([type], optional): Provide to replace with your custom "episode end here calculator". When not given, use environments' default done calculator. Defaults to None.
            max_observations (list, optional): TOMODIFY: Defaults to [1.0, 1.0].
            min_observations (list, optional): TOMODIFY: Defaults to [-1.0, -1.0].
            max_actions (list, optional): TOMODIFY: Defaults to [1.0, 1.0].
            min_actions (list, optional): TOMODIFY: Defaults to [-1.0, -1.0].
            observation_name (list, optional): the name of each observation. Defaults to None.
            action_name (list, optional): the name of each action. Defaults to None.
            initial_state_deviation_ratio (float, optional): the ratio of the initial state deviation to use for sample_initial_state. Defaults to None.
            np_dtype (type, optional): Defaults to np.float32.
            max_steps (int, optional): TOMODIFY: Defaults to None.
            error_reward (float, optional): When an error is encountered during an episode (this typically means something really bad, like tank overflow). Defaults to -100.0.
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
        self.initial_state_deviation_ratio = initial_state_deviation_ratio
        self.np_dtype = np_dtype
        self.max_steps = max_steps
        self.error_reward = error_reward

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

    def observation_beyond_box(self, observation):
        """check if the observation is beyond the box, which is what we don't want.

        Args:
            observation ([np.ndarray]): This is denormalized observation, as usual.

        Returns:
            [bool]: observation is beyond the box or not.
        """
        #TODO: check for how long?
        observation = self.np_dtype(observation)
        return (not self.observation_space.contains(observation)) or np.any(
            np.isnan(observation)) or np.any(np.isinf(observation))
        # return np.any(observation > self.max_observations) or np.any(observation < self.min_observations) or np.any(
        #     np.isnan(observation)) or np.any(np.isinf(observation))
        
    def action_beyond_box(self, action):
        """check if the action is beyond the box, which is what we don't want.

        Args:
            action ([np.ndarray]): This is denormalized action, as usual.

        Returns:
            [bool]: action is beyond the box or not.
        """       
        action = self.np_dtype(action)
        return (not self.action_space.contains(action)) or np.any(
            np.isnan(action)) or np.any(np.isinf(action))
        # return np.any(action > self.max_actions) or np.any(action < self.min_actions) or np.any(
        #     np.isnan(action)) or np.any(np.isinf(action))

    def reward_function_standard(self, previous_observation, action, current_observation, reward=None):
        """the s, a, r, s, a calculation.

        Args:
            previous_observation ([np.ndarray]): This is denormalized observation, as usual.
            action ([np.ndarray]): This is denormalized action, as usual.
            current_observation ([np.ndarray]): This is denormalized observation, as usual.
            reward ([float]): If reward is provided, directly return the reward.

        Returns:
            [float]: reward.
        """
        if self.observation_beyond_box(current_observation) or self.action_beyond_box(action):
            return self.error_reward
        elif reward is not None:
            # TOMODIFY: insert your own reward function here.
            reward = reward
        
        reward = max(self.error_reward, reward)  # reward cannot be smaller than the error_reward
        if self.debug_mode:
            print("reward:", reward)
        return reward

    def done_calculator_standard(self, current_observation, step_count, reward, done=None, done_info=None):
        """check whether the current episode is considered finished.
            returns a boolean value indicated done or not, and a dictionary with information.
            here in done_calculator_standard, done_info looks like {"terminal": boolean, "timeout": boolean},
            where "timeout" is true when episode end due to reaching the maximum episode length,
            "terminal" is true when "timeout" or episode end due to termination conditions such as env error encountered. (basically done)
            
        Args:
            current_observation ([np.ndarray]): This is denormalized observation, as usual.
            step_count ([int]): step_count.
            reward ([float]): reward.
            done ([bool], optional): Defaults to None.
            done_info ([dict], optional): how the environment is finished. Defaults to None.

        Returns:
            [(float, dict)]: done and done_info.
        """
        if done is None:
            done = False
        else:
            if done_info is not None:
                return done, done_info
            else:
                raise Exception("When done is given, done_info should also be given.")

        if done_info is None:
            done_info = {"terminal": False, "timeout": False}
            # done_info["terminal"] and done_info["timeout"] can only be flip from False to True, not vice versa.
        else:
            if done_info["terminal"] or done_info["timeout"]:
                done = True
                return done, done_info
        
        # check for valid observation
        if self.observation_beyond_box(current_observation): 
            # here we dont have action_beyond_box since action should not be passed in to done_calculator_standard.
            # however if action is beyond the box in step fucntion, reward_function_standard has already checked it
            # and the error_reward check below will be triggered.
            done_info["terminal"] = True
            done = True
        # check for valid reward
        if reward == self.error_reward:
            done_info["terminal"] = True
            done = True
        if math.isnan(reward):
            done_info["terminal"] = True
            done = True
        # check for step_count
        if step_count >= self.max_steps:  # same as range(0, max_steps)
            done_info["terminal"] = True
            done_info["timeout"] = True
            done = True
            
        # TOMODIFY: insert your own done calculator here.

        return done, done_info
    
    def observation_done_and_reward_calculator(self, current_observation, action, normalize=None, step_reward=None, done_info=None):
        """the s, a, r, s, a rollout, with error checks.

        Args:
            current_observation (list or np.ndarray): This is denormalized observation, as usual.
            previous_observation (np.ndarray): This is denormalized observation, as usual.
            action (np.ndarray): This is denormalized action, as usual.
            normalize (bool): Defaults to None.
            step_reward (float, optional): The reward of current step. Defaults to None.
            done_info (dict, optional): Defaults to None.
        
        Returns:
            observation (np.ndarray): This is the returned observation controlled by the normalize argument, for step function.
            [(float, bool, dict)]: reward, done and done_info.
            done_info looks like {"timeout": boolean, "error_occurred": boolean, "terminal": boolean},
            where "timeout" is true when episode end due to reaching the maximum episode length,
            "error_occurred": is true when episode end due to env error encountered,
            "terminal" is true when "timeout" or episode end due to termination conditions such as product collection is finished. (basically done).
            "terminal" should be True whenever timeout or error_occurred.
        """
        current_observation = np.array(current_observation, dtype=self.np_dtype)
        
        if done_info is None:
            done_info = {"timeout": False, "error_occurred": False, "terminal": False}
            # done_info["terminal"] and done_info["timeout"] can only be flip from False to True, not vice versa.
        
        # check for step_count
        if self.step_count >= self.max_steps:  # same as range(0, max_steps)
            done_info["terminal"] = True
            done_info["timeout"] = True
        # check for valid observation and action
        if self.observation_beyond_box(current_observation) or self.action_beyond_box(action):
            done_info["terminal"] = True
            done_info["error_occurred"] = True
        # check for valid reward
        if step_reward == self.error_reward:
            done_info["terminal"] = True
            done_info["error_occurred"] = True
        if math.isnan(step_reward):
            done_info["terminal"] = True
            done_info["error_occurred"] = True
        
        # TOMODIFY: mostly you won't, but you can add your own done calculation here if necessary.
        
        if not done_info["terminal"]:
            # TOMODIFY: compute your reward here, or use the reward argument passed in by step function.
            step_reward = step_reward
        
        # check for valid reward, again.
        step_reward = max(self.error_reward, step_reward)  # reward cannot be smaller than the error_reward
        if step_reward == self.error_reward:
            done_info["terminal"] = True
            done_info["error_occurred"] = True
        if math.isnan(step_reward):
            done_info["terminal"] = True
            done_info["error_occurred"] = True
        
        if done_info["error_occurred"] or done_info["timeout"]:
            assert done_info["terminal"]

        if done_info["error_occurred"] is True: # error occured. Set reward to self.error_reward.
            self.total_reward = self.error_reward
            reward = self.error_reward
        else: # figure out step_reward or total_reward to return.
            self.total_reward += step_reward
            if self.dense_reward:
                reward = step_reward
            elif not done_info["terminal"]:
                reward = 0.0
            else:
                reward = self.total_reward
        
        if self.debug_mode:
            print("reward:", reward)
            
        observation = current_observation
        self.previous_observation = observation
        observation = observation.clip(self.min_observations, self.max_observations) # clip observation so that it won't be beyond the box
        if normalize:
            observation, _, _ = normalize_spaces(observation, self.max_observations, self.min_observations)
        self.step_count += 1
         
        return observation, reward, done_info["terminal"], done_info

    def sample_initial_state(self):
        return self.observation_space.sample()

    def reset(self, initial_state=None, normalize=None):
        """
        Required by gym, this function resets the environment and returns an initial observation.
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
        
        normalize = self.normalize if normalize is None else normalize
        if normalize:
            observation, _, _ = normalize_spaces(observation, self.max_observations, self.min_observations)
        return observation


    def step(self, action, normalize=None):
        """
        Required by gym, his function performs one step within the environment and returns the observation, the reward, whether the episode is finished and debug information, if any.
        """
        if self.debug_mode:
            print("action:", action)
        reward = None
        done_info = None
        action = np.array(action, dtype=self.np_dtype)
        normalize = self.normalize if normalize is None else normalize
        if normalize:
            action, _, _ = denormalize_spaces(action, self.max_actions, self.min_actions)

        try:
            # TOMODIFY: proceed your environment here and collect the observation. The observation should be a numpy array.
            # optionally, done or done_info cane be provided here.
            # If done = True, then done_info["terminal"] and at least one of done_info["timeout"] or done_info["error_occurred"] should be True.
            observation = [0.0, 0.0]
            # done_info = {"timeout": boolean, "error_occurred": boolean, "terminal": boolean}
        except Exception as e:
            observation = self.min_observations
            done_info = {"timeout": False, "error_occurred": True, "terminal": True}
        
        observation, reward, done, done_info = self.observation_done_and_reward_calculator(observation, action, normalize=normalize, step_reward=reward, done_info=done_info)
        info = {}
        info.update(done_info)
        return observation, reward, done, info


    def set_initial_states(self, initial_states, num_episodes):
        if initial_states is None:
            initial_states = [self.sample_initial_state() for _ in range(num_episodes)]
        elif isinstance(initial_states, str):
            initial_states = np.load(initial_states)
        assert len(initial_states) == num_episodes
        return initial_states
    
    def evenly_spread_initial_states(self, val_per_state, dump_location=None):
        """
        Evenly spread initial states.
        This function is needed only if the environment has steady_observations. 
        
        Args:
            val_per_state (int): how many values to sampler per state.
            
        Returns:
        [initial_states]: evenly spread initial_states.
        """
        initial_state_deviation_ratio = self.initial_state_deviation_ratio
        steady_observations = self.steady_observations
        len_obs = len(steady_observations)
        val_range = val_per_state ** len_obs
        initial_states = np.zeros([val_range, len_obs])
        tmp_o = []
        for oi in range(len_obs):
            tmp_o.append(np.linspace(steady_observations[oi] * (1.0 - initial_state_deviation_ratio),
                                     steady_observations[oi] * (1.0 + initial_state_deviation_ratio), num=val_per_state,
                                     endpoint=True))

        for i in range(val_range):
            tmp_val_range = i
            curr_val = []
            for oi in range(len_obs):
                rmder = tmp_val_range % val_per_state
                curr_val.append(tmp_o[oi][rmder])
                tmp_val_range = int((tmp_val_range - rmder) / val_per_state)
            initial_states[i] = curr_val
        if dump_location is not None:
            np.save(dump_location, initial_states)
        return initial_states


    def evalute_algorithms(self, algorithms, num_episodes=1, error_reward=None, initial_states=None, to_plt=True,
                           plot_dir='./plt_results'):
        """
        when excecuting evalute_algorithms, the self.normalize should be False.
        algorithms: list of (algorithm, algorithm_name, normalize). algorithm has to have a method predict(observation) -> action: np.ndarray.
        num_episodes: number of episodes to run
        error_reward: overwrite self.error_reward
        initial_states: None, location of numpy file of initial states or a (numpy) list of initial states
        to_plt: whether generates plot or not
        plot_dir: None or directory to save plots
        returns: observations_list, actions_list, rewards_list
        """
        try:
            assert self.normalize is False
        except AssertionError:
            print("env.normalize should be False when executing evalute_algorithms")
            self.normalize = False
        if error_reward is not None:
            self.error_reward = error_reward
        if plot_dir is not None:
            mkdir_p(plot_dir)
        initial_states = self.set_initial_states(initial_states, num_episodes)
        observations_list = [[] for _ in range(
            len(algorithms))]  # observations_list[i][j][t][k] is algorithm_i_game_j_observation_t_element_k
        actions_list = [[] for _ in
                        range(len(algorithms))]  # actions_list[i][j][t][k] is algorithm_i_game_j_action_t_element_k
        rewards_list = [[] for _ in range(len(algorithms))]  # rewards_list[i][j][t] is algorithm_i_game_j_reward_t
        for n_epi in tqdm(range(num_episodes)):
            for n_algo in range(len(algorithms)):
                algo, algo_name, normalize = algorithms[n_algo]
                try:
                    algo.reset()
                except AttributeError:
                    pass
                algo_observes = []
                algo_actions = []
                algo_rewards = []  # list, for this algorithm, reawards of this trajectory.
                try:
                    init_obs = self.reset(initial_state=initial_states[n_epi])
                except TypeError: # means env cant set with initial_state, like PenSimEnvGym
                    init_obs = self.reset()
                # algo_observes.append(init_obs)
                o = init_obs
                done = False
                while not done:
                    if normalize:
                        o, _, _ = normalize_spaces(o, self.max_observations, self.min_observations)
                    a = algo.predict(o)
                    if normalize:
                        a, _, _ = denormalize_spaces(a, self.max_actions, self.min_actions)
                    algo_actions.append(a)
                    o, r, done, _ = self.step(a)
                    algo_observes.append(o)
                    algo_rewards.append(r)
                observations_list[n_algo].append(algo_observes)
                actions_list[n_algo].append(algo_actions)
                rewards_list[n_algo].append(algo_rewards)

            if to_plt:
                # plot observations
                for n_o in range(self.observation_dim):
                    o_name = self.observation_name[n_o]

                    plt.close("all")
                    plt.figure(0)
                    plt.title(f"{o_name}")
                    for n_algo in range(len(algorithms)):
                        alpha = 1 * (0.7 ** (len(algorithms) - 1 - n_algo))
                        _, algo_name, _ = algorithms[n_algo]
                        plt.plot(np.array(observations_list[n_algo][-1])[:, n_o], label=algo_name, alpha=alpha)
                    plt.plot([initial_states[n_epi][n_o] for _ in range(self.max_steps)], linestyle="--",
                             label=f"initial_{o_name}")
                    plt.xticks(np.arange(1, self.max_steps + 2, 1))
                    plt.annotate(str(initial_states[n_epi][n_o]), xy=(0, initial_states[n_epi][n_o]))
                    plt.legend()
                    if plot_dir is not None:
                        path_name = os.path.join(plot_dir, f"{n_epi}_observation_{o_name}.png")
                        plt.savefig(path_name)
                    plt.close()

                # plot actions
                for n_a in range(self.action_dim):
                    a_name = self.action_name[n_a]

                    plt.close("all")
                    plt.figure(0)
                    plt.title(f"{a_name}")
                    for n_algo in range(len(algorithms)):
                        alpha = 1 * (0.7 ** (len(algorithms) - 1 - n_algo))
                        _, algo_name, _ = algorithms[n_algo]
                        plt.plot(np.array(actions_list[n_algo][-1])[:, n_a], label=algo_name, alpha=alpha)
                    plt.xticks(np.arange(1, self.max_steps + 2, 1))
                    plt.legend()
                    if plot_dir is not None:
                        path_name = os.path.join(plot_dir, f"{n_epi}_action_{a_name}.png")
                        plt.savefig(path_name)
                    plt.close()

                # plot rewards
                plt.close("all")
                plt.figure(0)
                plt.title("reward")
                for n_algo in range(len(algorithms)):
                    alpha = 1 * (0.7 ** (len(algorithms) - 1 - n_algo))
                    _, algo_name, _ = algorithms[n_algo]
                    plt.plot(np.array(rewards_list[n_algo][-1]), label=algo_name, alpha=alpha)
                plt.xticks(np.arange(1, self.max_steps + 2, 1))
                plt.legend()
                if plot_dir is not None:
                    path_name = os.path.join(plot_dir, f"{n_epi}_reward.png")
                    plt.savefig(path_name)
                plt.close()

        observations_list = np.array(observations_list)
        actions_list = np.array(actions_list)
        rewards_list = np.array(rewards_list)
        return observations_list, actions_list, rewards_list
    
    
    def algorithms_to_algo_names(self, algorithms):
        """

        Args:
            algorithms: list of (algorithm, algorithm_name, normalize).
            
        Returns:
            list of algorithm_name.
        """
        algo_names = []
        for algo, algo_name, _ in algorithms:
            algo_names.append(algo_name)
        return algo_names


    def report_rewards(self, rewards_list, algo_names=None, save_dir=None):
        """
        returns: mean and std of rewards over all episodes.
        since the rewards_list is not aligned (e.g. some trajectories are shorter than the others), so we cannot directly convert it to numpy array.
        we have to convert and unwrap the nested list.
        on_episodes first average the rewards_list over episodes, then compute the mean and std.
        all_rewards directly compute the mean and std for each step.
        # rewards_list[i][j][t] is algorithm_i_game_j_reward_t.
        """
        result_dict = {}
        if algo_names is None:
            algo_names = []
            for i in range(len(rewards_list)):
                algo_names.append(f'algo_{i}')
        num_episodes = len(rewards_list[0])
        for n_algo in range(len(algo_names)):
            algo_name = algo_names[n_algo]
            rewards_list_curr_algo = rewards_list[n_algo]
            rewards_mean_over_episodes = []  # rewards_mean_over_episodes[n_epi] is mean of rewards of n_epi
            for n_epi in range(num_episodes):
                if rewards_list_curr_algo[n_epi][-1] == self.error_reward:
                    rewards_mean_over_episodes.append(self.error_reward)
                else:
                    rewards_mean_over_episodes.append(np.mean(rewards_list_curr_algo[n_epi]))
            on_episodes_reward_mean = np.mean(rewards_mean_over_episodes)
            on_episodes_reward_std = np.std(rewards_mean_over_episodes)
            unwrap_list = []
            for games_r_list in rewards_list_curr_algo:
                unwrap_list += list(games_r_list)
            all_reward_mean = np.mean(unwrap_list)
            all_reward_std = np.std(unwrap_list)
            print(f"{algo_name}_on_episodes_reward_mean: {on_episodes_reward_mean}")
            result_dict[algo_name + "_on_episodes_reward_mean"] = on_episodes_reward_mean
            print(f"{algo_name}_on_episodes_reward_std: {on_episodes_reward_std}")
            result_dict[algo_name + "_on_episodes_reward_std"] = on_episodes_reward_std
            print(f"{algo_name}_all_reward_mean: {all_reward_mean}")
            result_dict[algo_name + "_all_reward_mean"] = all_reward_mean
            print(f"{algo_name}_all_reward_std: {all_reward_std}")
            result_dict[algo_name + "_all_reward_std"] = all_reward_std
        mkdir_p(save_dir)
        f_dir = os.path.join(save_dir, 'result.json')
        json.dump(result_dict, open(f_dir, 'w+'))
        return result_dict
        
    def dataset_to_observations_actions_rewards_list(self, dataset):
        """_summary_

        Args:
            dataset (_type_): d4rl or torch format dataset obtained from generate_dataset_with_algorithm
            
        Returns:
            the same as evalute_algorithms
        """
        try:
            dataset = TorchDatasetFromD4RL(dataset) # convert to torch dataset
        except TypeError:
            pass
        observations_list_0 = []
        actions_list_0 = []
        rewards_list_0 = []
        # observations_list[i][j][t][k] is algorithm_i_game_j_observation_t_element_k
        for j in range(len(dataset)):
            curr_game = dataset[j]
            observations_list_0.append(curr_game['observations'].tolist())
            actions_list_0.append(curr_game['actions'].tolist())
            rewards_list_0.append(curr_game['rewards'].tolist())
        observations_list = [observations_list_0]
        actions_list = [actions_list_0]
        rewards_list = [rewards_list_0]
        return observations_list, actions_list, rewards_list

    def generate_dataset_with_algorithm(self, algorithm, normalize=None, num_episodes=1, error_reward=-1000.0,
                                        initial_states=None, format='d4rl'):
        """
        this function aims to create a dataset for offline reinforcement learning, in either d4rl or pytorch format.
        the trajectories are generated by the algorithm, which interacts with this env initialized by initial_states.
        algorithm: an instance that has a method predict(observation) -> action: np.ndarray.
        if format == 'd4rl', returns a dictionary in d4rl format.
        else if format == 'torch', returns an object of type torch.utils.data.Dataset.
        """
        if normalize is None:
            normalize = self.normalize
        initial_states = self.set_initial_states(initial_states, num_episodes)
        dataset = {}
        dataset["observations"] = []
        dataset["actions"] = []
        dataset["rewards"] = []
        dataset["terminals"] = []
        dataset["timeouts"] = []
        dataset["predict_time_taken"] = []
        for n_epi in tqdm(range(num_episodes)):
            o = self.reset(initial_state=initial_states[n_epi])
            r = 0.0
            done = False
            timeout = False
            final_done = False  # to still record for the last t when done
            while not final_done:
                if done:
                    final_done = True
                # tmp_o is to be normalized, if normalize is true.
                tmp_o = o
                if normalize:
                    tmp_o, _, _ = normalize_spaces(tmp_o, self.max_observations, self.min_observations)
                curr_time = time.time()
                a = algorithm.predict(tmp_o)
                time_taken = time.time() - curr_time
                curr_time = time.time()
                if normalize:
                    a, _, _ = denormalize_spaces(a, self.max_actions, self.min_actions)
                dataset['observations'].append(o)
                dataset['actions'].append(a)
                dataset['rewards'].append(r)
                dataset['terminals'].append(done)
                dataset["timeouts"].append(timeout)
                dataset['predict_time_taken'].append(time_taken)

                o, r, done, info = self.step(a)
                timeout = info['timeout']
        dataset["observations"] = np.array(dataset["observations"])
        dataset["actions"] = np.array(dataset["actions"])
        dataset["rewards"] = np.array(dataset["rewards"])
        dataset["terminals"] = np.array(dataset["terminals"])
        dataset["timeouts"] = np.array(dataset["timeouts"])
        if format == 'd4rl':
            return dataset
        elif format == 'torch':
            return TorchDatasetFromD4RL(dataset)
        else:
            raise ValueError(f"format {format} is not supported.")
