# -*- coding: utf-8 -*-
"""
AtropineEnv simulates an atropine production environment.
"""

import matplotlib.pyplot as plt
import numpy as np
from casadi import DM
from gym import spaces, Env
import requests
import io

from .helpers.constants import USS, INPUT_REFS, OUTPUT_REF, SIM_TIME
from .helpers.helper_funcs import state_estimator
from .models.atropine_process import Plant, atropine_mpc_controller
from .models.config import ND1, ND2, ND3, V1, V2, V3, V4, dt
from .utils import *


class AtropineMPC:
    def __init__(self, 
        model_loc='https://github.com/Quarticai/QuarticGym/blob/master/quarticgym/datasets/atropineenv/model.npy?raw=true',
        N=30, Nx=2, Nu=4,
        uss_subtracted=True, reward_on_ess_subtracted=False, reward_on_steady=True,
        reward_on_absolute_efactor=False, reward_on_actions_penalty=0.0, reward_on_reject_actions=True,
        relaxed_max_min_actions=False, observation_include_t=True, observation_include_action=False,
        observation_include_uss=True, observation_include_ess=True, observation_include_e=True,
        observation_include_kf=True, observation_include_z=True, observation_include_x=False):
        response = requests.get(model_loc)
        response.raise_for_status()
        self.model_preconfig = np.load(io.BytesIO(response.content), allow_pickle=True)
        self.N = N
        self.Nx = Nx
        self.Nu = Nu
        self.uss_subtracted = uss_subtracted  # we assume that we can see the steady state output during steps. If true, we plus the actions with USS during steps.
        self.reward_on_ess_subtracted = reward_on_ess_subtracted
        self.reward_on_steady = reward_on_steady  # whether reward base on Efactor (the small the better) or base on how close it is to the steady e-factor
        self.reward_on_absolute_efactor = reward_on_absolute_efactor  # whether reward base on absolute Efactor. (is a valid input only if reward_on_steady is False)
        self.reward_on_actions_penalty = reward_on_actions_penalty
        self.reward_on_reject_actions = reward_on_reject_actions  # when input actions are larger than max_actions, reject it and end the env immediately.
        self.relaxed_max_min_actions = relaxed_max_min_actions  # assume uss_subtracted = false.

        # now, select what to include during observations. by default we should have format like 
        # USS1, USS2, USS3, USS4, U1, U2, U3, U4, ESS, E, KF_X1, KF_X2, Z1, Z2, ..., Z30
        self.observation_include_t = observation_include_t  # 1
        self.observation_include_action = observation_include_action  # 4
        self.observation_include_uss = observation_include_uss  # 4
        self.observation_include_ess = observation_include_ess  # yss, 1
        self.observation_include_e = observation_include_e  # y, efactor, 1
        self.observation_include_kf = observation_include_kf  # after kalman filter, 2
        self.observation_include_z = observation_include_z  # 30
        self.observation_include_x = observation_include_x  # 1694
        assert (self.observation_include_kf)
        
    def predict(self, state):
        # self.observation_include_t = observation_include_t  # 1
        # self.observation_include_action = observation_include_action  # 4
        # self.observation_include_uss = observation_include_uss  # 4
        # self.observation_include_ess = observation_include_ess  # yss, 1
        # self.observation_include_e = observation_include_e  # y, efactor, 1
        # self.observation_include_kf = observation_include_kf  # after kalman filter, 2
        # self.observation_include_z = observation_include_z  # 30
        # self.observation_include_x = observation_include_x  # 1694
        observations = list(state)
        if self.observation_include_t:
            t = observations[:1]
            observations = observations[1:]
        if self.observation_include_action:
            action = observations[:4]
            observations = observations[4:]
        if self.observation_include_uss:
            uss = observations[:4]
            observations = observations[4:]
        if self.observation_include_ess:
            yss = observations[:1]
            observations = observations[1:]
        if self.observation_include_e:
            efactor = observations[:1]
            observations = observations[1:]
        if self.observation_include_kf:
            kf = observations[:2]
            observations = observations[2:]
        if self.observation_include_z:
            z = observations[:30]
            observations = observations[30:]
        if self.observation_include_x:
            x = observations[:1694]
            observations = observations[1694:]
        # TODO: self.model_preconfig diff from self.model.A, B, C, D, K !!!!!!!!!!
        uk = atropine_mpc_controller(
            np.array(kf),
            self.N, self.Nx, self.Nu,
            USS, INPUT_REFS, OUTPUT_REF,
            self.model_preconfig[0], self.model_preconfig[1],
            self.model_preconfig[2], self.model_preconfig[3],
        ) / 1000  # unscale
        if self.uss_subtracted is False:
            uk = uk + USS
        return uk


class AtropineEnvGym(smplEnvBase):
    def __init__(self, dense_reward=True, normalize=True, debug_mode=False, action_dim=4,
                 reward_function=None, done_calculator=None, 
                 observation_name=None, action_name=None, np_dtype=np.float32, max_steps=60, 
                 error_reward=-100000.0,
                 x0_loc='https://raw.githubusercontent.com/smpl-env/smpl/main/smpl/datasets/atropineenv/x0.txt',
                 z0_loc='https://raw.githubusercontent.com/smpl-env/smpl/main/smpl/datasets/atropineenv/z0.txt',
                 model_loc='https://github.com/Quarticai/QuarticGym/blob/master/quarticgym/datasets/atropineenv/model.npy?raw=true',
                 uss_subtracted=True, reward_on_ess_subtracted=False, reward_on_steady=True,
                 reward_on_absolute_efactor=False, reward_on_actions_penalty=0.0, reward_on_reject_actions=True, reward_scaler=1.0,
                 relaxed_max_min_actions=False, observation_include_t=True, observation_include_action=False,
                 observation_include_uss=True, observation_include_ess=True, observation_include_e=True,
                 observation_include_kf=True, observation_include_z=True, observation_include_x=False,
                #  init_mpc_controller=False
                 ):
        # define arguments
        self.step_count = 0
        self.total_reward = 0
        self.done = False
        self.dense_reward = dense_reward
        self.normalize = normalize  
        self.debug_mode = debug_mode  
        self.action_dim = action_dim
        self.reward_function = reward_function  
        self.done_calculator = done_calculator  
        self.observation_name = observation_name
        self.action_name = action_name
        self.np_dtype = np_dtype
        self.max_steps = max_steps # note that if self.max_steps == -1 then this simulator runs forever (open-loop).
        self.error_reward = error_reward
        if self.reward_function is None:
            self.reward_function = self.reward_function_standard
        if self.done_calculator is None:
            self.done_calculator = self.done_calculator_standard
            
        self.uss_subtracted = uss_subtracted  # we assume that we can see the steady state output during steps. If true, we plus the actions with USS during steps.
        self.reward_on_ess_subtracted = reward_on_ess_subtracted
        self.reward_on_steady = reward_on_steady  # whether reward base on Efactor (the small the better) or base on how close it is to the steady e-factor
        self.reward_on_absolute_efactor = reward_on_absolute_efactor  # whether reward base on absolute Efactor. (is a valid input only if reward_on_steady is False)
        self.reward_on_actions_penalty = reward_on_actions_penalty
        self.reward_on_reject_actions = reward_on_reject_actions  # when input actions are larger than max_actions, reject it and end the env immediately.
        self.reward_scaler = reward_scaler # a scaler multiplied on rewards to avoid numerical errors. By default, it is 1.0. In our paper, it is 100000.0 (which takes readability into account).
        self.relaxed_max_min_actions = relaxed_max_min_actions  # assume uss_subtracted = false.

        # now, select what to include during observations. by default we should have format like 
        # USS1, USS2, USS3, USS4, U1, U2, U3, U4, ESS, E, KF_X1, KF_X2, Z1, Z2, ..., Z30
        self.observation_include_t = observation_include_t  # 1
        self.observation_include_action = observation_include_action  # 4
        self.observation_include_uss = observation_include_uss  # 4
        self.observation_include_ess = observation_include_ess  # yss, 1
        self.observation_include_e = observation_include_e  # y, efactor, 1
        self.observation_include_kf = observation_include_kf  # after kalman filter, 2
        self.observation_include_z = observation_include_z  # 30
        self.observation_include_x = observation_include_x  # 1694
        
        if type(x0_loc) is str:
            response = requests.get(x0_loc)
            response.raise_for_status()
            self.x0 = np.loadtxt(io.BytesIO(response.content))
        elif type(x0_loc) is np.ndarray:
            self.x0 = x0_loc
        elif type(x0_loc) is list:
            self.x0 = np.array(x0_loc)
        else:
            raise Exception("x0_loc must be a string, list or a numpy array")
        if type(z0_loc) is str:
            response = requests.get(z0_loc)
            response.raise_for_status()
            self.z0 = np.loadtxt(io.BytesIO(response.content))
        elif type(z0_loc) is np.ndarray:
            self.z0 = z0_loc
        elif type(z0_loc) is list:
            self.z0 = np.array(z0_loc)
        else:
            raise Exception("z0_loc must be a string, list or a numpy array")
        response = requests.get(model_loc)
        response.raise_for_status()
        self.model_preconfig = np.load(io.BytesIO(response.content), allow_pickle=True)
        
        # for a fixed batch.
        self.ur = INPUT_REFS  # reference inputs
        self.yr = OUTPUT_REF  # reference output
        self.num_sim = int(SIM_TIME / dt)  # SIM_TIME/ 400 hours as fixed batch.

        self.observation_dim = 1 * self.observation_include_t + 4 * self.observation_include_action + 4 * self.observation_include_uss + 1 * self.observation_include_ess + \
                               1 * self.observation_include_e + 2 * self.observation_include_kf + 30 * self.observation_include_z + 1694 * self.observation_include_x
        
        if self.observation_name is None:
            self.observation_name = [f'o_{i}' for i in range(self.observation_dim)]
        if self.action_name is None:
            self.action_name = [f'a_{i}' for i in range(self.action_dim)]
        
        max_observations = []
        if self.observation_include_t:
            max_observations.append(np.ones(1, dtype=np.float32) * 100.0)  # by convention
        if self.observation_include_action:
            max_observations.append(np.ones(4, dtype=np.float32) * 1.0)  # by convention
        if self.observation_include_uss:
            max_observations.append(np.ones(4, dtype=np.float32) * 0.5)  # from dataset
        if self.observation_include_ess:
            max_observations.append(np.ones(1, dtype=np.float32) * 15.0)  # from dataset
        if self.observation_include_e:
            max_observations.append(np.ones(1, dtype=np.float32) * 20.0)  # from dataset
        if self.observation_include_kf:
            max_observations.append(np.ones(2, dtype=np.float32) * 5.0)  # from dataset
        if self.observation_include_z:
            max_observations.append(np.ones(30, dtype=np.float32) * 0.5)  # by convention
        if self.observation_include_x:
            max_observations.append(np.ones(1694, dtype=np.float32) * 50.0)  # by convention
        try:
            self.max_observations = np.concatenate(max_observations)
        except ValueError:
            raise Exception("observations must contain something! Need at least one array to concatenate")
        min_observations = []
        if self.observation_include_t:
            min_observations.append(np.ones(1, dtype=np.float32) * 0.0)  # by convention
        if self.observation_include_action:
            min_observations.append(np.ones(4, dtype=np.float32) * 0.0)  # by convention
        if self.observation_include_uss:
            min_observations.append(np.ones(4, dtype=np.float32) * 0.0)  # from dataset
        if self.observation_include_ess:
            min_observations.append(np.ones(1, dtype=np.float32) * 0.0)  # from dataset
        if self.observation_include_e:
            min_observations.append(np.ones(1, dtype=np.float32) * 0.0)  # from dataset
        if self.observation_include_kf:
            min_observations.append(np.ones(2, dtype=np.float32) * -5.0)  # from dataset
        if self.observation_include_z:
            min_observations.append(np.ones(30, dtype=np.float32) * 0.0)  # by convention
        if self.observation_include_x:
            min_observations.append(np.ones(1694, dtype=np.float32) * 0.0)  # by convention                       
        try:
            self.min_observations = np.concatenate(min_observations)
        except ValueError:
            raise Exception("observations must contain something! Need at least one array to concatenate")
        if not self.uss_subtracted:
            self.max_actions = np.array([0.408, 0.125, 0.392, 0.214], dtype=np.float32)  # from dataset
            self.min_actions = np.array([0.4075, 0.105, 0.387, 0.208], dtype=np.float32)  # from dataset
            if self.relaxed_max_min_actions:
                self.max_actions = np.array([0.5, 0.2, 0.5, 0.4], dtype=np.float32)  # from dataset
                self.min_actions = np.array([0.3, 0.0, 0.2, 0.1], dtype=np.float32)  # from dataset
        else:
            self.max_actions = np.array([1.92476206e-05, 1.22118426e-02, 1.82154982e-03, 3.59729230e-04],
                                        dtype=np.float32)  # from dataset
            self.min_actions = np.array([-0.00015742, -0.00146234, -0.00021812, -0.00300454],
                                        dtype=np.float32)  # from dataset
            if self.relaxed_max_min_actions:
                self.max_actions = np.array([2.0e-05, 1.3e-02, 2.0e-03, 4.0e-04], dtype=np.float32)  # from dataset
                self.min_actions = np.array([-0.00016, -0.0015, -0.00022, -0.00301], dtype=np.float32)  # from dataset
        if self.normalize:
            self.observation_space = spaces.Box(low=-1, high=1, shape=(self.observation_dim,))
            self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,))
        else:
            self.observation_space = spaces.Box(low=self.min_observations, high=self.max_observations,
                                                shape=(self.observation_dim,))
            self.action_space = spaces.Box(low=self.min_actions, high=self.max_actions, shape=(self.action_dim,))
        self.plant = Plant(ND1, ND2, ND3, V1, V2, V3, V4, dt)
        self.yss = self.plant.calculate_Efactor(DM(self.z0))  # steady state output, e-factor
        
    def sample_initial_state(self, no_sample=False, lower_bound=0.99, upper_bound=1.01):
        if no_sample:
            x0 = self.x0
        else:
            x0_c = self.x0.clip(1e-30) # treat all negative concentration values as numerical error and set to close to zero to avoid numerical issues.
            low = x0_c * lower_bound
            up = x0_c * upper_bound
            x0 = np.random.uniform(low, up)
        self.zk = DM(self.z0)  # shape is (30,)
        self.xk = self.plant.mix_and_get_initial_condition(x0, USS)[0]  # shape is (1694,)
        self.previous_efactor = self.yss  # steady state output, e-factor, the small the better
        observations = []
        if self.observation_include_t:
            observations.append(np.array([0], dtype=np.float32)) # since initial state should have self.step_count = 0
        if self.observation_include_action:
            observations.append(np.zeros(4, dtype=np.float32))
        if self.observation_include_uss:
            observations.append(USS)
        if self.observation_include_ess:
            observations.append(np.array([self.yss], dtype=np.float32))
        if self.observation_include_e:
            observations.append(np.array([self.previous_efactor], dtype=np.float32))
        if self.observation_include_kf:
            self.xe = np.ones(2, dtype=np.float32) * 0.001
            observations.append(self.xe)
        if self.observation_include_z:
            observations.append(self.zk.full().flatten())
        if self.observation_include_x:
            observations.append(self.xk.full().flatten())
        try:
            observation = np.concatenate(observations)
        except ValueError:
            raise Exception("observations must contain something! Need at least one array to concatenate")
        return observation
        
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
        self.U = []  # inputs
        self.Y = []
        
        normalize = self.normalize if normalize is None else normalize
        if normalize:
            observation, _, _ = normalize_spaces(observation, self.max_observations, self.min_observations)
        return observation
    
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
            reward = reward * self.reward_scaler
        
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
        if step_count >= self.max_steps and self.max_steps != -1:  # same as range(0, max_steps)
            done_info["terminal"] = True
            done_info["timeout"] = True
            done = True

        return done, done_info
    
    def _step(self, action, normalize=None):
        """
        Required by gym, his function performs one step within the environment and returns the observation, the reward, whether the episode is finished and debug information, if any.
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

        # TOMODIFY: proceed your environment here and collect the observation. The observation should be a numpy array.
        if self.uss_subtracted:
            uk = action
            uk_p_USS = [
                action[0] + USS[0],
                action[1] + USS[1],
                action[2] + USS[2],
                action[3] + USS[3]
            ]
        else:
            uk = action
            uk_p_USS = action
        self.U.append(uk_p_USS)
        _, xnext, znext = self.plant.simulate(self.xk, self.zk, uk_p_USS)
        efactor = self.plant.calculate_Efactor(znext)
        self.Y.append(efactor)
        # ---- reward calculation here, to avoid large if statements. ----
        reward_on_steady = -abs(efactor - self.yss)
        reward_on_absolute_efactor = -abs(efactor)
        reward_on_efactor_diff = self.previous_efactor - efactor
        previous_efactor = self.previous_efactor
        if self.reward_on_ess_subtracted:
            reward = self.yss - efactor  # efactor the smaller the better
        elif self.reward_on_steady:
            reward = reward_on_steady
        else:
            if self.reward_on_absolute_efactor:
                reward = reward_on_absolute_efactor
            else:
                reward = reward_on_efactor_diff
        reward += np.linalg.norm(action * self.reward_on_actions_penalty, ord=2)
        # ---- reward calculation here, to avoid large if statements. ----
        self.previous_efactor = efactor
        self.xk = xnext
        self.zk = znext
        observations = []
        if self.observation_include_t:
            observations.append(np.array([self.step_count], dtype=np.float32))
        if self.observation_include_action:
            observations.append(action)
        if self.observation_include_uss:
            observations.append(USS)
        if self.observation_include_ess:
            observations.append(np.array([self.yss], dtype=np.float32))
        if self.observation_include_e:
            observations.append(np.array([self.previous_efactor],
                                         dtype=np.float32))  # !!!!!!!also, shall I run a step here? how do we align?
        if self.observation_include_kf:
            self.xe = state_estimator(
                self.xe, uk, efactor - self.yss,  # self.Xhat[k] is previous step xe
                self.model_preconfig[0], self.model_preconfig[1],
                self.model_preconfig[2], self.model_preconfig[4]
            )
            observations.append(self.xe)
        if self.observation_include_z:
            observations.append(self.zk.full().flatten())
        if self.observation_include_x:
            observations.append(self.xk.full().flatten())
        try:
            observation = np.concatenate(observations)
        except ValueError:
            raise Exception("observations must contain something! Need at least one array to concatenate")

        observation = np.array(observation, dtype=self.np_dtype)
        # compute reward
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
        info = {
            "efactor": efactor, 
            "previous_efactor": previous_efactor,
            "reward_on_steady": reward_on_steady,
            "reward_on_absolute_efactor": reward_on_absolute_efactor,
            "reward_on_efactor_diff": reward_on_efactor_diff
            }
        info.update(done_info)
        return observation, reward, done, info

    def step(self, action, normalize=None):
        try:
            return self._step(action)
        except Exception:
            reward = self.error_reward
            done = True
            done_info = {"terminal": True, "timeout": False}
            observation = np.zeros(self.observation_dim, dtype=np.float32)
            info = {
                "efactor": -self.error_reward, 
                "previous_efactor": self.previous_efactor,
                "reward_on_steady": reward, 
                "reward_on_absolute_efactor": reward,
                "reward_on_efactor_diff": reward
            }
            return observation, reward, done, info

    def plot(self, show=False, efactor_fig_name=None, input_fig_name=None):
        target_efactor = [self.yss + self.yr] * self.num_sim
        target_inputs = [USS + self.ur] * self.num_sim
        U = np.array(self.U) * 1000  # scale the solution to micro Litres
        target_inputs = np.array(target_inputs) * 1000  # scale the solution to micro Litres
        local_t = [k * dt for k in range(self.num_sim)]
        # plots
        plt.close("all")
        plt.figure(0)
        plt.plot(local_t, self.Y, label='Real Output')
        plt.plot(local_t, target_efactor, linestyle="--", label='Steady State Output')
        plt.xlabel('Time [min]')
        plt.ylabel('E-Factor [A.U.]')
        plt.legend()
        plt.grid()
        if efactor_fig_name is not None:
            plt.savefig(efactor_fig_name)
        plt.tight_layout()

        # create figure (fig), and array of axes (ax)
        fig, axs = plt.subplots(nrows=2, ncols=2)
        axs[0, 0].step(local_t, U[:, 0], where='post', label='Real Input')
        axs[0, 0].step(local_t, target_inputs[:, 0], where='post', linestyle="--", label='Steady State Input')
        axs[0, 0].set_ylabel(u'U1 [\u03bcL/min]')
        axs[0, 0].set_xlabel('time [min]')
        axs[0, 0].grid()

        axs[0, 1].step(local_t, U[:, 1], where='post', label='Real Input')
        axs[0, 1].step(local_t, target_inputs[:, 1], where='post', linestyle="--", label='Steady State Input')
        axs[0, 1].set_ylabel(u'U2 [\u03bcL/min]')
        axs[0, 1].set_xlabel('time [min]')
        axs[0, 1].grid()

        axs[1, 0].step(local_t, U[:, 2], where='post', label='Real Input')
        axs[1, 0].step(local_t, target_inputs[:, 2], where='post', linestyle="--", label='Steady State Input')
        axs[1, 0].set_ylabel(u'U3 [\u03bcL/min]')
        axs[1, 0].set_xlabel('time [min]')
        axs[1, 0].grid()

        axs[1, 1].step(local_t, U[:, 3], where='post', label='Real Input')
        axs[1, 1].step(local_t, target_inputs[:, 3], where='post', linestyle="--", label='Steady State Input')
        axs[1, 1].set_ylabel(u'U4 [\u03bcL/min]')
        axs[1, 1].set_xlabel('time [min]')
        axs[1, 1].legend()
        plt.tight_layout()
        plt.grid()
        if input_fig_name is not None:
            plt.savefig(input_fig_name)
        if show:
            plt.show()
        else:
            plt.close()

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
                    # plt.plot([self.steady_observations[n_o] for _ in range(self.max_steps)], linestyle="-.",
                    #          label=f"steady_{o_name}") # add back when we confirm steady_observations yss+
                    plt.xticks(np.arange(1, self.max_steps + 2, 1))
                    plt.annotate(str(initial_states[n_epi][n_o]), xy=(0, initial_states[n_epi][n_o]))
                    # plt.annotate(str(self.steady_observations[n_o]), xy=(0, self.steady_observations[n_o])) # add back when we confirm steady_observations
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
                    # plt.plot([self.steady_actions[n_a] for _ in range(self.max_steps)], linestyle="-.",
                    #          label=f"steady_{a_name}") # add back when we confirm steady_actions USS+
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
