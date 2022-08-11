This repo contains configuration information and some example trajectories of environments.

AtropineEnv
-----------

The files are mainly ODE-model configurations. If you don't want to change the ODEs of the AtropineEnv, you should not change them.

initial state configuration of the atropine environment. The z0.txt has a dimensionality of 30, and the x0.txt has a dimensionality of 1694. z0.txt represents the states that are monitorable by sensors (flow rates, etc.) x0.txt are mostly intermediate states that are not directly visible or very costly to make visible. U.mat and model.npy are used by the Atropine model (the ODEs).


mAbEnv
------

Again, the files are mainly ODE model configurations. If you don't want to change the ODEs of the mAbEnv, you should not change them.
The ulb.npy, uub.npy are the lower and upper bounds of the mAb model (the ODEs) outputs. The xlb.npy and xub.npy are the lower and upper bounds of the mAb model inputs. the uss.npy and xss.npy are the steady outputs and inputs. The steady_action.npy and steady_observations.npy of the mAbEnv (the Gym-like environment for control algorithms to interact with).

PenSimEnv
---------

There are two example batches for the use of examples.ipynb. The full 1010 example Gaussian Process trajectories are in `smpl-experiments <https://github.com/smpl-env/smpl-experiments/tree/main/pensimenv_experiments/pensimpy_1010_samples>`_

