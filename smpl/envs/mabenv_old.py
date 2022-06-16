from .utils import *
import mpctools as mpc
from scipy import integrate
from .helpers.mab_helpers_old import xscale, uscale, UpModelHelper, DownModelHelper, UtilsHelper


class MAbUpstreamMPC:
    def __init__(
            self, action_dim=7, observation_dim=17,
            dt=1, ss_dir=None) -> None:
        
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.dt = dt
        self.upstream = UpModelHelper(xscale, uscale, self.observation_dim, self.action_dim, dt)
        self.utils_helper = UtilsHelper()
        self.xss, self.uss = self.utils_helper.load_ss(res_dir=ss_dir)
        Q = np.eye(self.observation_dim)
        R = np.eye(self.action_dim)
        N = 300
        self.mpc_cont = self.upstream.build_mpc_controller(self.xss, self.uss, Q, R, N, dt)
        # change tolerance
        solveroptions = {'tol': 1E-5}
        # change ipopt.linear_solver to ma27 for faster and robust computation. This needs to be installed on your system
        casadioptions = {'ipopt.linear_solver': 'mumps',
                            'verbose': False, 'ipopt.print_level': 0, 'ipopt.tol': 1E-4}
        self.mpc_cont.initialize(casadioptions=casadioptions, solveroptions=solveroptions)
        
    def predict(self, o):
        self.mpc_cont.fixvar("x", 0, o)
        self.mpc_cont.solve()
        if self.mpc_cont.stats["status"] != "Solve_Succeeded":
            0  # break
        else:
            self.mpc_cont.saveguess()
        ukm = np.squeeze(self.mpc_cont.var["u", 0])  # ukm is mpc's action
        return ukm
    
class MAbUpstreamEMPC:
    def __init__(
            self, action_dim=7, observation_dim=17,
            dt=1, ss_dir=None) -> None:
        
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.dt = dt
        self.upstream = UpModelHelper(xscale, uscale, self.observation_dim, self.action_dim, dt)
        self.utils_helper = UtilsHelper()
        self.xss, self.uss = self.utils_helper.load_ss(res_dir=ss_dir)
        Q = np.eye(self.observation_dim)
        R = np.eye(self.action_dim)
        N = 300
        # change tolerance
        solveroptions = {'tol': 1E-5}
        # change ipopt.linear_solver to ma27 for faster and robust computation. This needs to be installed on your system
        casadioptions = {'ipopt.linear_solver': 'mumps',
                            'verbose': False, 'ipopt.print_level': 0, 'ipopt.tol': 1E-4}
        # initialize empc
        self.empc_cont = self.upstream.build_empc_controller(self.xss, self.uss, N, dt)
        self.empc_cont.initialize(casadioptions=casadioptions, solveroptions=solveroptions)
        
    def predict(self, o):
        
        self.empc_cont.fixvar("x", 0, o)
        self.empc_cont.solve()
        if self.empc_cont.stats["status"] != "Solve_Succeeded":
            0  # break
        else:
            self.empc_cont.saveguess()

        uke = np.squeeze(self.empc_cont.var["u", 0])
        return uke

class MAbUpstreamEnvGym(smplEnvBase):
    def __init__(
            self, dense_reward=True, normalize=True, debug_mode=False, action_dim=7, observation_dim=17,
            reward_function=None, done_calculator=None,
            max_observations=[200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0,
                              200.0, 200.0, 200.0, 200.0],
            min_observations=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            max_actions=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            min_actions=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            observation_name=None, action_name=None, np_dtype=np.float32, max_steps=200, error_reward=-100.0,
            dt=1, init_mpc_controllers=False, ss_dir=None, standard_reward_style='setpoint') -> None:
        """[summary]

        Args:
            dt (int, optional): Time sampling (hr). Defaults to 1.
            init_mpc_controllers (bool, optional): Initialize MPC and EMPC controllers. Defaults to True.
            ss_dir (str, optional): Directory of steady state and steady action files. Defaults to None.
            standard_reward_style (str, optional): Reward style, can be 'setpoint' or 'productivity'. 
                The 'setpoint' reward bases on how the controller is able to move the observation close to
                the steady state observation; the 'productivity' reward bases on the MAb upstream productivity.
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

        self.dt = dt
        self.upstream = UpModelHelper(xscale, uscale, self.observation_dim, self.action_dim, dt)
        self.utils_helper = UtilsHelper()
        self.xss, self.uss = self.utils_helper.load_ss(res_dir=ss_dir)
        self.steady_observations = self.xss / xscale
        self.steady_actions  = self.uss / uscale
        self.plant = mpc.DiscreteSimulator(self.upstream.xdot_scale, self.dt, [
                                           self.observation_dim, self.action_dim], ["x", "u"])
        Q = np.eye(self.observation_dim)
        R = np.eye(self.action_dim)
        N = 300
        # initialize mpc
        self.init_mpc_controllers = init_mpc_controllers
        if self.init_mpc_controllers:
            self.mpc_cont = self.upstream.build_mpc_controller(self.xss, self.uss, Q, R, N, dt)
            # change tolerance
            solveroptions = {'tol': 1E-5}
            # change ipopt.linear_solver to ma27 for faster and robust computation. This needs to be installed on your system
            casadioptions = {'ipopt.linear_solver': 'mumps',
                             'verbose': False, 'ipopt.print_level': 0, 'ipopt.tol': 1E-4}
            self.mpc_cont.initialize(casadioptions=casadioptions, solveroptions=solveroptions)
            # initialize empc
            self.empc_cont = self.upstream.build_empc_controller(self.xss, self.uss, N, dt)
            self.empc_cont.initialize(casadioptions=casadioptions, solveroptions=solveroptions)
        self.standard_reward_style = standard_reward_style

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
        if reward is not None:
            return reward
        elif self.observation_beyond_box(current_observation) or self.action_beyond_box(action):
            return self.error_reward

        # TOMODIFY: insert your own reward function here.
        if self.standard_reward_style == 'setpoint':
            reward = (np.square(current_observation - self.steady_observations)).mean()
        elif self.standard_reward_style == 'productivity':
            xx = current_observation * xscale
            uu = action * uscale
            reward = xx[6] * uu[1] + current_observation[14] * uu[3]
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

    def sample_initial_state(self, lower_bound=0.95, upper_bound=1.05):
        """[summary]

        Args:
            lower_bound (float, optional): proportional to steady state. Defaults to 0.95.
            upper_bound (float, optional): proportional to steady state. Defaults to 1.05.

        Returns:
            [np.ndarray]: [description]
        """        """"""
        low = self.xss/xscale * lower_bound
        up = self.xss/xscale * upper_bound
        return np.random.uniform(low, up)

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
        self.Xm_mycontrol = []
        self.Um_mycontrol = []
        self.t = []
        self.Xm = []
        self.Um = []
        self.Xe = []
        self.Ue = []
        self.Xs = []
        self.Us = []
        # initial condition
        self.Xm_mycontrol += [self.init_observation]
        self.Xm += [self.init_observation]
        self.Xe += [self.init_observation]
        self.Xs += [self.xss]

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
        self.t += [self.step_count*self.dt]
        if self.init_mpc_controllers:
            self.mpc_cont.fixvar("x", 0, self.Xm[self.step_count])
            self.mpc_cont.solve()
            print("%5s %d: %s" % ("MPC", self.t[self.step_count], self.mpc_cont.stats["status"]))
            if self.mpc_cont.stats["status"] != "Solve_Succeeded":
                0  # break
            else:
                self.mpc_cont.saveguess()
            ukm = np.squeeze(self.mpc_cont.var["u", 0])  # ukm is mpc's action
            # ---- to capture numpy warnings ---- 
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("error")
                try:
                    xkm = self.plant.sim(self.Xm[self.step_count], ukm)  # xkm is mpc's next observation
                except Exception as e:
                    print("Got MPC Exception/Warning: ", e)
                    xkm = self.Xm[self.step_count]
            # /---- to capture numpy warnings ---- 
            self.Um += [ukm]
            self.Xm += [xkm]

            self.empc_cont.fixvar("x", 0, self.Xe[self.step_count])
            self.empc_cont.solve()

            # Print status and make sure solver didn't fail.
            print("%5s %d: %s" % ("EMPC", self.step_count, self.empc_cont.stats["status"]))
            if self.empc_cont.stats["status"] != "Solve_Succeeded":
                0  # break
            else:
                self.empc_cont.saveguess()

            uke = np.squeeze(self.empc_cont.var["u", 0])
            # uk = uss

            # advance one time step
            # ---- to capture numpy warnings ---- 
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("error")
                try:
                    xke = self.plant.sim(self.Xe[self.step_count], uke)  # xkm is empc's next observation
                except Exception as e:
                    print("Got EMPC Exception/Warning: ", e)
                    xke = self.Xe[self.step_count]
            # /---- to capture numpy warnings ---- 

            # store solution
            self.Ue += [uke]
            self.Xe += [xke]

        # for plotting
        self.Us += [self.uss]
        self.Xs += [self.xss]

        # ---- to capture numpy warnings ---- 
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("error")
            try:
                observation = self.plant.sim(self.Xm_mycontrol[self.step_count], action) # self.plant.sim returns an observation within range [0, 200]
            except Exception as e:
                print("Got Exception/Warning: ", e)
                observation = self.previous_observation
                reward = self.error_reward
                done = True
                done_info["terminal"] = True
        # /---- to capture numpy warnings ---- 
        self.Um_mycontrol += [action]
        self.Xm_mycontrol += [observation]

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
