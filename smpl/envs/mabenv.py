import copy
from .utils import *
import mpctools as mpc
from scipy import integrate
import timeout_decorator
from .helpers.mab_helpers import xscale, uscale, UtilsHelper, ControllerHelper
STEP_TIMEOUT_LENGTH = 30 # how long in second(s) a step can take before a timeout error is triggered.

class MAbUpstreamMPC:
    def __init__(
            self, controller, action_dim=7+1+1, observation_dim=17+2+1951) -> None:
        
        self.controller = controller
        # self.xss = xss
        # self.uss = uss
        # self.xscale = xscale
        # self.uscale = uscale
        # self.dt_spl = dt_spl
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        num_x = [17, 2, 1951]  # Number of states for each unit
        num_u = [7, 1, 1]      # Number of inputs for each unit
        Q = np.eye(num_x[0]); R = np.eye(num_u[0]); N = 300 # Q, R, N are for the EMPC controller. N is the memory length.
        solver_opts = {'tol': 1E-5}  # Change tolerance
        casadi_opts = {'ipopt.linear_solver': 'mumps', 'verbose': False, 'ipopt.print_level': 0, 'ipopt.tol': 1E-4}
        self.mpc_cont = controller._build_mpc_up(Q, R, N, controller.dt_spl, controller.xss[:controller.Nx_up], controller.uss[:controller.Nu_up], controller.xscale[:controller.Nx_up], controller.uscale[:controller.Nu_up])
        controller.mpc_cont = self.mpc_cont
        self.mpc_cont.initialize(casadioptions=casadi_opts, solveroptions=solver_opts)
        
    def predict(self, o):
        u = np.zeros(self.action_dim)
        x = o
        x_up = x[0:17]  # 17
        x_buffer = x[17:19]  # 2
        x_down = x[19:]  # 1951
        self.mpc_cont.fixvar("x", 0, x_up)
        self.mpc_cont.solve()
        if self.mpc_cont.stats["status"] != "Solve_Succeeded":
            0  # break
        else:
            self.mpc_cont.saveguess()
        u[:7] = np.squeeze(self.mpc_cont.var["u", 0])
        u[7] = self.controller._pcontroller(x_buffer[0])
        u[8] = self.controller._switcher(x_down)
        return u
    
class MAbUpstreamEMPC:
    def __init__(
            self, controller, action_dim=7+1+1, observation_dim=17+2+1951) -> None:
        
        self.controller = controller
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        num_x = [17, 2, 1951]  # Number of states for each unit
        num_u = [7, 1, 1]      # Number of inputs for each unit
        Q = np.eye(num_x[0]); R = np.eye(num_u[0]); N = 300 # Q, R, N are for the EMPC controller. N is the memory length.
        solver_opts = {'tol': 1E-5}  # Change tolerance
        casadi_opts = {'ipopt.linear_solver': 'mumps', 'verbose': False, 'ipopt.print_level': 0, 'ipopt.tol': 1E-4}
        self.empc_cont = controller._build_empc_up(N, controller.dt_spl, controller.xss[:controller.Nx_up], controller.uss[:controller.Nu_up], controller.xscale[:controller.Nx_up], controller.uscale[:controller.Nu_up])
        controller.empc_cont = self.empc_cont
        self.empc_cont.initialize(casadioptions=casadi_opts, solveroptions=solver_opts)
        
    def predict(self, o):
        u = np.zeros(self.action_dim)
        x = o
        x_up = x[0:17]  # 17
        x_buffer = x[17:19]  # 2
        x_down = x[19:]  # 1951
        self.empc_cont.fixvar("x", 0, x_up)
        self.empc_cont.solve()
        if self.empc_cont.stats["status"] != "Solve_Succeeded":
            0  # break
        else:
            self.empc_cont.saveguess()
        u[:7] = np.squeeze(self.empc_cont.var["u", 0])
        u[7] = self.controller._pcontroller(x_buffer[0])
        u[8] = self.controller._switcher(x_down)
        return u


class MAbEnvGym(smplEnvBase):
    def __init__(
            self, dataset_dir='smpl/datasets/mabenv', dense_reward=True, normalize=True, debug_mode=False, action_dim=7+1+1, observation_dim=17+2+1951,
            reward_function=None, done_calculator=None,
            max_observations=None,
            min_observations=None,
            max_actions=None,
            min_actions=None,
            observation_name=None, action_name=None, np_dtype=np.float32, max_steps=200, error_reward=-100.0, initial_state_deviation_ratio=0.1,
            upstream_states=17+2, switch_threshold=0.5, dt_itgr=60, dt_spl=60, ss_dir=None, standard_reward_style='setpoint') -> None:
        """[summary]

        Args:
            dataset_dir (str, optional): The dataset directory that has uss.npy, xss.npy, ulb.npy, uub.npy,
                xlb.npy, xub.npy. You could find it in 'smpl/datasets/mabenv' when you clone from github.
            upstream_states  (int, optional): The number of states to use for the upstream.
            initial_state_deviation_ratio  (float, optional): The initial state range around steady states. Defaults to 0.1.
            switch_threshold (float, optional): When action[-1] >= switch_threshold, we change the buffer tank.
                Defaults to 0.5.
            dt_itgr (int, optional): Time integration (min), ode solver dt_itgr per step.
            dt_spl (int, optional): Time sampling (min) sample a observation from the model every dt_spl.
            init_mpc_controllers (bool, optional): Initialize MPC and EMPC controllers. Defaults to True.
            ss_dir (str, optional): Directory of steady state and steady action files. Defaults to None.
            standard_reward_style (str, optional): Reward style, can be 'setpoint', 'productivity' or 'yield'. 
                The 'setpoint' reward bases on how the controller is able to move the observation close to
                the steady state observation; the 'productivity' reward bases on the MAb upstream productivity;
                the 'yield' computes the collected mAb yield from downstream.
                Defaults to 'setpoint'.
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
            self.observation_name = ['Xv1', 'Xt1', 'GLC1', 'GLN1', 'LAC1', 'AMM1', 'mAb1', 'V1', 'Xv2', 'Xt2', 'GLC2', 'GLN2', 'LAC2', 'AMM2', 'mAb2', 'V2', 'T']
        if self.action_name is None:
            self.action_name = ['F_in', 'F_1', 'F_r', 'F_2', 'GLC_in', 'GLN_in', 'Tc']
        self.np_dtype = np_dtype
        self.max_steps = max_steps
        self.error_reward = error_reward
        if self.reward_function is None:
            self.reward_function = self.reward_function_standard
        if self.done_calculator is None:
            self.done_calculator = self.done_calculator_standard

        self.upstream_states = upstream_states
        self.initial_state_deviation_ratio = initial_state_deviation_ratio
        self.switch_threshold = switch_threshold
        self.dataset_dir = dataset_dir
        self.dt_itgr = dt_itgr
        self.dt_spl = dt_spl
        self.standard_reward_style = standard_reward_style
        self.num_x = [17, 2, 1951]  # Number of states for each unit
        self.num_u = [7, 1, 1]      # Number of inputs for each unit
        assert sum(self.num_x) == observation_dim
        assert sum(self.num_u) == action_dim
        
        self.utils_helper = UtilsHelper()
        self.xss, self.uss = self.utils_helper.prepare_ss(self.dataset_dir)  # 9, 1970
        self.steady_observations = self.xss / xscale
        self.steady_actions  = self.uss / uscale
        # set max and min
        self.min_actions, self.max_actions, self.min_observations, self.max_observations = self.utils_helper.load_bounds(self.dataset_dir) 
        self.min_actions, self.max_actions, self.min_observations, self.max_observations = self.min_actions / uscale, self.max_actions / uscale, self.min_observations / xscale, self.max_observations / xscale
        self.controller = ControllerHelper(self.num_x, self.num_u, self.max_steps, dt_itgr, dt_spl, xscale, uscale, self.xss, self.uss)
        self.plant = self.controller._build_plant()
        
        
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
        return np.any(observation[:self.upstream_states] > self.max_observations[:self.upstream_states]*1.05) or np.any(observation[:self.upstream_states] < self.min_observations[:self.upstream_states]) or np.any(
            np.isnan(observation)) or np.any(np.isinf(observation))

    def reward_function_standard(self, previous_observation, action, current_observation, reward=None):
        if reward is not None:
            return reward
        elif self.observation_beyond_box(current_observation) or self.action_beyond_box(action):
            return self.error_reward

        # TOMODIFY: insert your own reward function here.
        if self.standard_reward_style == 'setpoint':
            reward = -(np.square(current_observation[:17+2] - self.steady_observations[:17+2])).mean()
        else:
            xx = current_observation * xscale
            uu = action * uscale
            productivity = xx[6] * uu[1] + current_observation[14] * uu[3] # range of reward is [0, inf)
            if self.standard_reward_style == 'productivity':
                reward = productivity
            elif self.standard_reward_style == 'yield':
                # current_observation[19] is the inlet concentration
                # current_observation[-14] is the outlet concentration. The smaller the fewer waste, the better.
                downstream_yield = 1 - current_observation[-14]/(current_observation[19]+1e-8) # range of downstream_yield is [0,1].
                reward = productivity * downstream_yield
            else:
                raise ValueError("standard_reward_style should be either 'setpoint' or 'productivity'")
            

        reward = max(self.error_reward, reward)  # reward cannot be smaller than the error_reward
        if self.debug_mode:
            print("reward:", reward)
        return reward

    def done_calculator_standard(self, current_observation, step_count, reward, done=None, done_info=None):
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

        return done, done_info

    def sample_initial_state(self, lower_bound=None, upper_bound=None):
        """[summary]

        Args:
            lower_bound (float, optional): proportional to steady state.
            upper_bound (float, optional): proportional to steady state.

        Returns:
            [np.ndarray]: [description]
        """
        self.upstream_states
        if lower_bound is None:
            lower_bound = 1 - self.initial_state_deviation_ratio
        if upper_bound is None:
            upper_bound = 1 + self.initial_state_deviation_ratio
        low = self.steady_observations[:self.upstream_states] * lower_bound
        low = np.concatenate([low,  self.steady_observations[self.upstream_states:]])
        up = self.steady_observations[:self.upstream_states] * upper_bound
        up = np.concatenate([up,  self.steady_observations[self.upstream_states:]])
        return np.random.uniform(low, up)
    
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
        steady_observations = self.steady_observations[:self.upstream_states]
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
            initial_states[i] = np.concatenate([initial_states[i],  self.steady_observations[self.upstream_states:]])
        if dump_location is not None:
            np.save(dump_location, initial_states)
        return initial_states

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
            observation = self.sample_initial_state()  # x0
            self.init_observation = observation
        self.previous_observation = observation

        # TOMODIFY: reset your environment here.
        self.t = []
        self.Xi = []
        self.Xi += [observation]

        normalize = self.normalize if normalize is None else normalize
        if normalize:
            observation, _, _ = normalize_spaces(observation, self.max_observations, self.min_observations)
        return observation
    
    @timeout_decorator.timeout(STEP_TIMEOUT_LENGTH)
    def _simulation(self, xk, uk):
        xk = copy.deepcopy(xk)
        uk = copy.deepcopy(uk)
        
        if uk[-1] >= self.switch_threshold:
            # if self.debug_mode:
            #     print(xk[-1] * uk[7], 'mg of mAb is captured. Switching the column')
            print(xk[-1] * uk[7], 'mg of mAb is captured. Switching the column')
            xk[19:-1] = 0  # (1950). Set state vector of downstream capture column to 0
            xk[-1] = 0  # In new column, accumulated mab is 0
            uk[-1] = 0  # Reset
        # Integrator
        xkp1 = self.plant.sim(xk, uk)
        # update accumulated mAb
        xkp1[-1] += self.dt_itgr * (
                    xkp1[19] - xkp1[-14])  # difference between inlet concentration and outlet concentration
        return xkp1

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
        self.t += [self.step_count*self.dt_spl]


        # ---- to capture numpy warnings ---- 
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("error")
            try:
                for i in range(0, self.controller.dt_ratio):
                    xk = self._simulation(self.Xi[self.step_count * self.controller.dt_ratio + i], action)
                    self.Xi += [copy.deepcopy(xk)]
                    observation = xk
            except Exception as e:
                print("Got Exception/Warning: ", e)
                observation = self.previous_observation
                reward = self.error_reward
                done = True
                done_info["terminal"] = True
        # /---- to capture numpy warnings ---- 

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
                try:
                    algo.reset()
                except AttributeError:
                    pass
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
