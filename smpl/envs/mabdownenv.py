from re import S
from .utils import *
import mpctools as mpc
from scipy import integrate
from .helpers.mab_helpers_old import xscale, uscale, UpModelHelper, DownModelHelper, UtilsHelper


"""
Note: this file is not ready and under continuous development
"""
class MAbDownstreamEnvGym(smplEnvBase):
    def __init__(self, dense_reward=True, normalize=True, debug_mode=False, action_dim=1, observation_dim=1952,
            reward_function=None, done_calculator=None,
            max_observations=[10000, 10]+[10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 36.45, 77.85]*150,
            min_observations=np.zeros(1952),
            max_actions=[1.0],
            min_actions=[-1.0],
            observation_name=None, action_name=None, np_dtype=np.float32, max_steps=200, error_reward=-100.0,
            
            num_z_cap_load = 150,  # discretization points in Z direction for capture column loading process
            num_z_cap_elu = 150,   # discretization points in Z direction for capture column elution process
            num_z_cex_load = 150,  # discretization points in Z direction for CEX loading process
            num_z_cex_elu = 150,   # discretization points in Z direction for CEX elution process
            num_z_loop = 150,      # discretization points in Z direction for loop process (VI and Holdup loop)
            num_z_aex = 150,       # discretization points in Z direction for AEX process
            num_r = 10,            # discretization points in r direction for capture process
            num_sim=int(6000*8),   #* 8*24 hrs.
            delta_t=0.01,          # (min), since the downstream model is stiff. A smaller dt is required
        ):
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
            
        self.num_z_cap_load = num_z_cap_load
        self.num_z_cap_elu = num_z_cap_elu
        self.num_z_cex_load = num_z_cex_load
        self.num_z_cex_elu = num_z_cex_elu
        self.num_z_loop = num_z_loop
        self.num_z_aex = num_z_aex
        self.num_r = num_r
        self.num_sim = num_sim
        self.delta_t = delta_t
        self.utils_helper = UtilsHelper()
        
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
            
    def reward_function_standard(self, previous_observation, action, current_observation, reward=None):
        r = self.compute_recover_rate(current_observation)  # input shape (2+1950,)
        return r
    
    def compute_recover_rate(self, x):  # Recovery yield = purified product/harvested mAb
        harvested = x[-3]*x[-2]*x[-1]
        bottom = np.max([20000*self.delta_t, x[-1]])
        purified = (20000*self.delta_t+bottom)*x[-2]/2*x[-3]
        return purified/harvested
    
    def done_calculator_standard(self, current_observation, step_count, reward, done=None, done_info=None):
        pass # TODO:
    
    def sample_initial_state(self, lower_bound=0.95, upper_bound=1.05):
        flow_from_upper = np.array([8333, 1]) # TODO:
        x0_d = np.zeros((self.num_z_cap_load*13,)) # TODO:
        return np.concatenate([flow_from_upper, x0_d])
    
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
        flow_from_upper = self.previous_observation[:2]
        multi_states = self.previous_observation[2:] # (self.num_z_cap_load*13,) = 1950
        self.downstream = DownModelHelper(num_z_cap_load=self.num_z_cap_load,
            num_z_cap_elu=self.num_z_cap_elu,
            num_z_cex_load=self.num_z_cex_load,
            num_z_cex_elu=self.num_z_cex_elu,
            num_z_loop=self.num_z_loop,
            num_z_aex=self.num_z_aex,
            num_r=self.num_r,
            num_sim=self.num_sim_d,
            delta_t=self.dt_d,
            init_states=multi_states,
            inputs=flow_from_upper
        )
        self.X_all = []
        self.X_d = []

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
        # there are two configurations
        # if action[0] == 1: # TODO: change after we got other controls.
        # action is now binarized.
        # if action[0] > 0: # TODO:!!!!!!!!!!! diff in config?
        model = self.downstream.capture_load
        flow_from_upper = self.previous_observation[:2]
        multi_states = self.previous_observation[2:] # (self.num_z_cap_load*13,) = 1950
        sol = integrate.solve_ivp(fun=lambda t, y: model(t, y, u=flow_from_upper), # self.u_all[i - 1] is flow_from_upper
                                    t_span=[0, self.delta_t], y0=tuple(multi_states)) # self.x_all[i - 1, :]) shape 1950
        xk = sol['y'][:, -1] # TODO: observation (1950,)
        flow_from_upper = flow_from_upper # TODO: change after we got other controls.
        observation = np.concatenate([flow_from_upper, xk])
        # else TODO: change after we got other controls.

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
        