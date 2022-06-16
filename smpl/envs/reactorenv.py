# -*- coding: utf-8 -*-
"""
ReactorEnv simulates a general reactor environment. This is supposed to be an template environment. The documentations in that file is enhanced and provided comment lines (# ---- standard ---- and # /---- standard ----) enclose pieces of code that should be reused by most of smpl environments. Please consult the smplEnvBase class.
"""

import json
import os
import random
# ---- to capture numpy warnings ----
import warnings

import matplotlib.pyplot as plt
import numpy as np
from gym import spaces, Env  # to create an openai-gym environment https://gym.openai.com/
from mzutils import SimplePriorityQueue
from scipy.integrate import solve_ivp  # the ode solver
from tqdm import tqdm
import torch
import mpctools as mpc

from .utils import *

np.seterr(all='warn')
# ---- to capture numpy warnings ---- 


# macros the defined by the reactor
MAX_OBSERVATIONS = [1.0, 100.0, 1.0]  # cA, T, h
MIN_OBSERVATIONS = [1e-08, 1e-08, 1e-08]
MAX_ACTIONS = [35.0, 0.2]  # Tc, qout
MIN_ACTIONS = [15.0, 0.05]
STEADY_OBSERVATIONS = [0.8778252, 51.34660837, 0.659]
STEADY_ACTIONS = [26.85, 0.1]
ERROR_REWARD = -1000.0


class ReactorPID:
    """a simple proportional controller that utilizes the config of this environment.

    Returns:
        [type]: action.
    """
    def __init__(self, Kis, steady_state=[0.8778252, 0.659], steady_action=[26.85, 0.1], min_action=[15.0, 0.05], max_action=[35.0, 0.2]):
        self.Kis = Kis
        self.steady_state = steady_state
        self.steady_action = steady_action
        self.len_c = len(steady_action)
        self.min_action = min_action
        self.max_action = max_action
    
    def predict(self, state):
        state = [state[0], state[2]]
        action = []
        for i in range(self.len_c):
            a = self.Kis[i] * (state[i] - self.steady_state[i]) + self.steady_action[i]
            action.append(np.clip(a, self.min_action[i], self.max_action[i]))
        return np.array(action)
    

class ReactorMPC:
    """a simple MPC controller.

    Returns:
        [type]: action.
    """
    def __init__(self, Nt=20, dt=0.1, Q=np.eye(3)*0.1, R=np.eye(2)*0.0, P=np.eye(3)*0.1):
        self.Nt = Nt    # control and prediction horizons
        self.dt = dt    # sampling time. Should be the same as the step size in the model
        self.Q = Q      # weight on the state variables
        self.R = R      # weight on the input variables
        self.P = P      # terminal cost weight (optional). set to zero matrix with appropriate dimension
        self.Nx = 3     # number of state variables
        self.Nu = 2     # number of input variables
        self.min_action=[15.0, 0.05]
        self.max_action=[35.0, 0.2]

        # reactor model
        from smpl.envs.reactorenv import ReactorModel
        self.model = ReactorModel(dt)
        
        self.xs = np.array([0.8778252,51.34660837,0.659])   # steady-state state values 
        self.us = np.array([26.85,0.1])     # steady state input values
        self.build_controller()
        
    # Define stage cost
    def lfunc(self, x, u):
        dx = x[:self.Nx] - self.xs[:self.Nx]
        du = u[:self.Nu] - self.us[:self.Nu]
        return mpc.mtimes(dx.T,self.Q,dx) + mpc.mtimes(du.T,self.R,du)
    
    # define terminal weight
    def Pffunc(self, x):
        dx = x[:self.Nx] - self.xs[:self.Nx]
        return mpc.mtimes(dx.T,self.P,dx)
    
    # build the mpc controller using mpctools
    def build_controller(self):
        
        # stage and terminal cost in casadi symbolic form
        l = mpc.getCasadiFunc(self.lfunc, [self.Nx,self.Nu], ["x","u"], funcname="l")
        Pf = mpc.getCasadiFunc(self.Pffunc, [self.Nx], ["x"], funcname="Pf")
        
        # Create casadi function/model to be used in the controller
        ode_casadi = mpc.getCasadiFunc(self.model.ode, [self.Nx, self.Nu], ["x","u"], funcname="odef")

        # Set c to a value greater than one to use collocation
        contargs = dict(
                N = {"t":self.Nt, "x":self.Nx, "u":self.Nu, "c":3},
                verbosity=0,    # set verbosity to a number > 0 for debugging purposes otherwise use 0.
                l=l,
                x0=self.xs,
                Pf=Pf,
                ub = {"u":self.max_action}, # Change upper bounds 
                lb = {"u":self.min_action},	# Change lower bounds 
                guess = {
                        "x":self.xs[:self.Nx],
                        "u":self.us[:self.Nu]
                        },
                )
        
        # create MPC controller
        self.mpc = mpc.nmpc(f=ode_casadi,Delta=self.dt,**contargs)
    
    # solve mpc optimization problem at one time step
    def predict(self,x):
        # Update intial condition and solve control problem.
        self.mpc.fixvar("x",0,x) # Set initial value
        self.mpc.solve() # solve optimal control problem
        uopt = np.squeeze(self.mpc.var["u",0]) # take the first optimal input solution
        
        # if successful, save solution as initial guess for the next optimization problem
        if self.mpc.stats["status"] == "Solve_Succeeded":
            self.mpc.saveguess()
        
        # return solution status and optimal input
        return uopt


class ReactorMLPReinforceAgent(torch.nn.Module):
    """a simple torch-MLP based rl agent,.

    Returns:
        [type]: Can either return a sampled action or the distribution the action sampled from.
    """
    def __init__(self, obs_dim=3, act_dim=2, hidden_size=6, device='cpu'):
        super(ReactorMLPReinforceAgent, self).__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = torch.nn.Sequential(torch.nn.Linear(obs_dim, hidden_size),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(hidden_size, act_dim))
        self.device = device
        
    def forward(self, obs, return_distribution=True):
        mu = self.mu_net(obs)
        std = torch.maximum(torch.exp(self.log_std), torch.ones_like(self.log_std)*1e-4)
        dist = torch.distributions.normal.Normal(mu, std)
        if return_distribution:
            return dist
        else:
            action = np.clip(dist.sample().numpy().flatten(), [0, 0], [100, 0.2])  # clip actions
            return action


class ReactorModel:

    def __init__(self, sampling_time):
        # Define reactor model parameters
        self.q_in = .1  # m^3/min
        self.Tf = 76.85  # degrees C
        self.cAf = 1.0  # kmol/m^3
        self.r = .219  # m
        self.k0 = 7.2e10  # min^-1
        self.E_divided_by_R = 8750  # K
        self.U = 54.94  # kg/min/m^2/K
        self.rho = 1000  # kg/m^3
        self.Cp = .239  # kJ/kg/K
        self.dH = -5e4  # kJ/kmol

        self.Nx = 3  # Number of state variables
        self.Nu = 2  # Number of input variables

        self.sampling_time = sampling_time  # sampling time or integration step

    # Ordinary Differential Equations (ODEs) described in the report i.e. Equations (1), (2), (3)
    def ode(self, x, u):
        c = x[0]  # c_A
        T = x[1]  # T
        h = x[2]  # h
        Tc = u[0]  # Tc
        q = u[1]  # q_out

        rate = self.k0 * c * np.exp(-self.E_divided_by_R / (T + 273.15))  # kmol/m^3/min

        dxdt = [
            self.q_in * (self.cAf - c) / (np.pi * self.r ** 2 * h + 1e-8) - rate,  # kmol/m^3/min
            self.q_in * (self.Tf - T) / (np.pi * self.r ** 2 * h + 1e-8)
            - self.dH / (self.rho * self.Cp) * rate
            + 2 * self.U / (self.r * self.rho * self.Cp) * (Tc - T),  # degree C/min
            (self.q_in - q) / (np.pi * self.r ** 2)  # m/min
        ]
        return dxdt

    # integrates one sampling time or time step and returns the next state
    def step(self, x, u):
        # return self.simulator.sim(x,u)
        sol = solve_ivp(lambda t, x, u: self.ode(x, u), [0, self.sampling_time], x, args=(u,), method='LSODA')
        return sol.y[:, -1]


class ReactorEnvGym(smplEnvBase):

    def __init__(self, dense_reward=True, normalize=True, debug_mode=False, action_dim=2, observation_dim=3,
                 reward_function=None, done_calculator=None, max_observations=MAX_OBSERVATIONS,
                 min_observations=MIN_OBSERVATIONS, max_actions=MAX_ACTIONS, min_actions=MIN_ACTIONS,
                 error_reward=ERROR_REWARD,  # general env inputs
                 initial_state_deviation_ratio=0.3, compute_diffs_on_reward=False, np_dtype=np.float32,
                 sampling_time=0.1, max_steps=100):
        # ---- standard ----
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
        self.error_reward = error_reward
        if self.reward_function is None:
            self.reward_function = self.reward_function_standard
        if self.done_calculator is None:
            self.done_calculator = self.done_calculator_standard
        # /---- standard ----

        self.compute_diffs_on_reward = compute_diffs_on_reward  # how the reward is computed, if True, then the reward is computed as the difference between the current state and the previous state
        self.np_dtype = np_dtype
        self.sampling_time = sampling_time
        self.max_steps = max_steps

        self.observation_name = ["cA", "T", "h"]
        self.action_name = ["Tc", "qout"]

        # ---- standard ----
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
        # /---- standard ----

        self.steady_observations = np.array(STEADY_OBSERVATIONS, dtype=self.np_dtype)  # cA, T, h
        self.steady_actions = np.array(STEADY_ACTIONS, dtype=self.np_dtype)  # Tc, qout
        self.initial_state_deviation_ratio = initial_state_deviation_ratio


    def reward_function_standard(self, previous_observation, action, current_observation, reward=None):
        # ---- standard ----
        # s, a, r, s, a
        if reward is not None:
            return reward
        elif self.observation_beyond_box(current_observation) or self.action_beyond_box(action):
            return self.error_reward
        # /---- standard ----
        current_observation_evaluated = self.evaluate_observation(current_observation)
        assert isinstance(current_observation_evaluated, float)
        if self.compute_diffs_on_reward:
            previous_observation_evaluated = self.evaluate_observation(previous_observation)
            assert isinstance(previous_observation_evaluated, float)
            reward = current_observation_evaluated - previous_observation_evaluated
        else:
            reward = current_observation_evaluated
        # ---- standard ----
        reward = max(self.error_reward, reward)  # reward cannot be smaller than the error_reward
        if self.debug_mode:
            print("reward:", reward)
        return reward
        # /---- standard ----


    def reset(self, initial_state=None, normalize=None):
        # ---- standard ----
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
        # /---- standard ----
        self.reactor = ReactorModel(self.sampling_time)

        # ---- standard ----
        normalize = self.normalize if normalize is None else normalize
        if normalize:
            observation, _, _ = normalize_spaces(observation, self.max_observations, self.min_observations)
        return observation
        # /---- standard ----

    def step(self, action, normalize=None):
        # ---- standard ----
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
        # /---- standard ----

        # ---- to capture numpy warnings ---- 
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("error")
            try:
                observation = self.reactor.step(self.previous_observation, action)
            except Exception as e:
                print("Got Exception/Warning: ", e)
                observation = self.previous_observation
                reward = self.error_reward
                done = True
                done_info["terminal"] = True
        # /---- to capture numpy warnings ---- 

        # ---- standard ----
        # compute reward
        if not reward:
            reward = self.reward_function(self.previous_observation, action, observation, reward=reward)
        # compute done
        if not done:
            done, done_info = self.done_calculator(observation, self.step_count, reward, done=done, done_info=done_info)
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
        info = {}
        info.update(done_info)
        return observation, reward, done, info
        # /---- standard ----

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

    # ---- standard ----
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
        returns: list of average_rewards over each episode and num of episodes
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
                algo_observes = []
                algo_actions = []
                algo_rewards = []  # list, for this algorithm, reawards of this trajectory.
                init_obs = self.reset(initial_state=initial_states[n_epi])
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
                    plt.plot([self.steady_observations[n_o] for _ in range(self.max_steps)], linestyle="-.",
                             label=f"steady_{o_name}")
                    plt.xticks(np.arange(1, self.max_steps + 2, 1))
                    plt.annotate(str(initial_states[n_epi][n_o]), xy=(0, initial_states[n_epi][n_o]))
                    plt.annotate(str(self.steady_observations[n_o]), xy=(0, self.steady_observations[n_o]))
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
                    plt.plot([self.steady_actions[n_a] for _ in range(self.max_steps)], linestyle="-.",
                             label=f"steady_{a_name}")
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
        # /---- standard ----

    def evaluate_rewards_mean_std_over_episodes(self, algorithms, num_episodes=1, error_reward=None,
                                                initial_states=None, to_plt=True, plot_dir='./plt_results',
                                                computer_on_episodes=False):
        """
        returns: mean and std of rewards over all episodes.
        since the rewards_list is not aligned (e.g. some trajectories are shorter than the others), so we cannot directly convert it to numpy array.
        we have to convert and unwrap the nested list.
        if computer_on_episodes, we first average the rewards_list over episodes, then compute the mean and std.
        else, we directly compute the mean and std for each step.
        """
        result_dict = {}
        observations_list, actions_list, rewards_list = self.evalute_algorithms(algorithms, num_episodes=num_episodes,
                                                                                error_reward=error_reward,
                                                                                initial_states=initial_states,
                                                                                to_plt=to_plt, plot_dir=plot_dir)
        from warnings import warn
        warn('The function evaluate_rewards_mean_std_over_episodes is deprecated. Please use report_rewards.', DeprecationWarning, stacklevel=2)
        for n_algo in range(len(algorithms)):
            _, algo_name, _ = algorithms[n_algo]
            rewards_list_curr_algo = rewards_list[n_algo]
            if computer_on_episodes:
                rewards_mean_over_episodes = []  # rewards_mean_over_episodes[n_epi] is mean of rewards of n_epi
                for n_epi in range(num_episodes):
                    if rewards_list_curr_algo[n_epi][-1] == self.error_reward: # if error_reward is provided, self.error_reward is overwritten in self.evalute_algorithms
                        rewards_mean_over_episodes.append(self.error_reward)
                    else:
                        rewards_mean_over_episodes.append(np.mean(rewards_list_curr_algo[n_epi]))
                rewards_mean = np.mean(rewards_mean_over_episodes)
                rewards_std = np.std(rewards_mean_over_episodes)
            else:
                unwrap_list = []
                for games_r_list in rewards_list_curr_algo:
                    unwrap_list += games_r_list
                rewards_mean = np.mean(unwrap_list)
                rewards_std = np.std(unwrap_list)
            print(f"{algo_name}_reward_mean: {rewards_mean}")
            result_dict[algo_name + "_reward_mean"] = rewards_mean
            print(f"{algo_name}_reward_std: {rewards_std}")
            result_dict[algo_name + "_reward_std"] = rewards_std
        if plot_dir is not None:
            f_dir = os.path.join(plot_dir, 'result.json')
        else:
            f_dir = 'result.json'
        json.dump(result_dict, open(f_dir, 'w+'))
        return observations_list, actions_list, rewards_list

    def find_outperformances(self, algorithms, rewards_list, initial_states, threshold=0.05, top_k=10):
        """
        this function computes the outperformances of the last algorithm in algorithms.
        there are three criteria:
        if in a trajectory, the algorithm has reward >= all other algorithms, the corresponding initial_state is stored to always_better.
        if in a trajectory, the algorithm's mean reward >= threshold + all other algorithms' mean reward, the corresponding initial_state is stored to averagely_better.
        for the top_k most outperformed reward mean, the corresponding initial_state is stored to top_k_better, in ascending order.
        """
        # rewards_list[i][j][t] is algorithm_i_game_j_reward_t
        num_episodes = len(rewards_list[0])
        num_algorithms = len(algorithms)
        assert num_algorithms == len(rewards_list)
        assert num_algorithms >= 2
        always_better = []
        averagely_better = []
        top_k_better = SimplePriorityQueue(maxsize=top_k)
        for n_epi in range(num_episodes):
            rewards = [rewards_list[n_algo][n_epi] for n_algo in range(num_algorithms)]
            if self.find_outperformances_compute_always_better(rewards):
                always_better.append(initial_states[n_epi])
            average_outperformance = max(self.find_outperformances_compute_average_outperformances(rewards))
            if average_outperformance >= threshold:
                averagely_better.append(initial_states[n_epi])
            # like indicated here, https://stackoverflow.com/questions/42236820/adding-numpy-array-to-a-heap-queue, inserting numpy array to heapq can be risky.
            try:
                top_k_better.put((average_outperformance, initial_states[n_epi]))
            except ValueError:
                top_k_better.put((average_outperformance + random.uniform(0, 1e-8), initial_states[n_epi]))
        always_better = np.array(always_better)
        averagely_better = np.array(averagely_better)
        top_k_better = np.array([a[1] for a in top_k_better.nlargest(top_k)])
        return always_better, averagely_better, top_k_better

    def find_outperformances_compute_always_better(self, rewards):
        num_algorithms = len(rewards)
        for t in range(len(rewards[-1])):
            for n_algo in range(num_algorithms - 1):
                try:
                    if rewards[-1][t] < rewards[n_algo][t]:
                        return False
                except IndexError:
                    # some algorithms might finish the trajectory earlier.
                    pass
        return True

    def find_outperformances_compute_average_outperformances(self, rewards):
        num_algorithms = len(rewards)
        average_rewards = []
        for n_algo in range(num_algorithms):
            if rewards[n_algo][-1] == self.error_reward:
                average_rewards.append(self.error_reward)
            else:
                average_rewards.append(np.mean(rewards[n_algo]))
        outperformances = []  # we can hereby use just a scalar, but to reuse the code further for other criteria, we use a list.
        # E.g. in the future we can add a random walk algorithm as our baseline to compare the relative improvement with.
        for n_algo in range(num_algorithms - 1):
            outperformances.append((average_rewards[-1] - average_rewards[n_algo]) / 1.0)
        return outperformances

    def sample_initial_state(self):
        init_observation = np.maximum(
            np.random.uniform(low=(1 - self.initial_state_deviation_ratio) * self.steady_observations,
                              high=(1 + self.initial_state_deviation_ratio) * self.steady_observations), 0,
            dtype=self.np_dtype)
        init_observation = init_observation.clip(self.min_observations, self.max_observations)
        return init_observation


    def evaluate_observation(self, observation):
        """
        observation: numpy array of shape (self.observation_dim)
        returns: observation evaluation (reward in a sense)
        """

        return float(- (np.mean((observation - self.steady_observations) ** 2 / np.maximum(
            (self.init_observation - self.steady_observations) ** 2, 1e-8))))


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
                a = algorithm.predict(tmp_o)
                if normalize:
                    a, _, _ = denormalize_spaces(a, self.max_actions, self.min_actions)
                dataset['observations'].append(o)
                dataset['actions'].append(a)
                dataset['rewards'].append(r)
                dataset['terminals'].append(done)
                dataset["timeouts"].append(timeout)

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
