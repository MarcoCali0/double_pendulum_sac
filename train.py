import os
import sys

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.sac.policies import MlpPolicy

import constants
from model.model_parameters import model_parameters
from model.symbolic_plant import SymbolicDoublePendulum
from simulation.gym_env import DoublePendulumEnv, double_pendulum_dynamics_func
from simulation.simulation import Simulator
from utils.reset_functions import noisy_reset_func, zero_reset_func

log_dir = f"log_data/SAC_training"

if not os.path.exists(log_dir):
    os.makedirs(log_dir)


# define robot variation
acrobot_flag = "--acrobot" in sys.argv
pendubot_flag = "--pendubot" in sys.argv

if (acrobot_flag and pendubot_flag) or not (acrobot_flag or pendubot_flag):
    print("Invalid input! Use either --pendubot or --acrobot")
    exit(1)

robot = "acrobot" if acrobot_flag else "pendubot"

print(f"Training {robot} using SAC")
# Set random seed to zero for reproducibility
set_random_seed(0)

max_velocity = constants.max_velocity
max_torque = constants.max_torque

torque_limit = [max_torque, 0.0] if robot == "pendubot" else [0.0, max_torque]
model_par_path = f"parameters/common_parameters.yml"

# Model parameters
mpar = model_parameters(filepath=model_par_path)
mpar.set_torque_limit(torque_limit)

print("Loading model parameters...")
dt = 0.01  # 100 Hz
max_steps = 1_500  #  15 s episode
integrator = "runge_kutta"

plant = SymbolicDoublePendulum(model_pars=mpar)
simulator = Simulator(plant=plant, integrator_name=integrator)

# learning environment parameters
state_representation = 2
obs_space = gym.spaces.Box(
    np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0])
)
act_space = gym.spaces.Box(np.array([-1]), np.array([1]))

# tuning parameter
n_envs = 20
training_steps = 15_000_000
verbose = 1
eval_freq = 5000
n_eval_episodes = 1
learning_rate = 0.005  # proposed 0.01, but default is 3e-4

print(
    f"Training settings: training step = {training_steps}, learning rate = {learning_rate}"
)
# initialize double pendulum dynamics
dynamics_func = double_pendulum_dynamics_func(
    simulator=simulator,
    dt=dt,
    integrator=integrator,
    robot=robot,
    state_representation=state_representation,
    max_velocity=max_velocity,
    torque_limit=torque_limit,
)


# initialize vectorized environment
env = DoublePendulumEnv(
    dynamics_func=dynamics_func,
    reset_func=noisy_reset_func,
    obs_space=obs_space,
    act_space=act_space,
    max_episode_steps=max_steps,
)

# training env
envs = make_vec_env(
    seed=0,
    env_id=DoublePendulumEnv,
    n_envs=n_envs,
    env_kwargs={
        "dynamics_func": dynamics_func,
        "reset_func": noisy_reset_func,
        "obs_space": obs_space,
        "act_space": act_space,
        "max_episode_steps": max_steps,
    },
)

# evaluation env, same as training env
eval_env = DoublePendulumEnv(
    dynamics_func=dynamics_func,
    reset_func=zero_reset_func,
    obs_space=obs_space,
    act_space=act_space,
    max_episode_steps=max_steps,
)


class CustomEvalCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super(CustomEvalCallback, self).__init__(*args, **kwargs)
        self.eval_count = 0

    def on_step(self) -> bool:
        result = super(CustomEvalCallback, self).on_step()
        if self.n_calls % self.eval_freq == 0:
            self.eval_count += 1
            model_path = os.path.join(self.log_path, f"{robot}_model_{self.eval_count}")
            self.model.save(model_path)
        return result


# training callbacks
eval_callback = CustomEvalCallback(
    eval_env,
    best_model_save_path=os.path.join(log_dir, f"{robot}_best_model"),
    log_path=log_dir,
    eval_freq=eval_freq,
    verbose=verbose,
    n_eval_episodes=n_eval_episodes,
)

# train
agent = SAC(
    MlpPolicy,
    envs,
    verbose=verbose,
    tensorboard_log=os.path.join(log_dir, f"logs_{robot}"),
    learning_rate=learning_rate,
)


agent.learn(total_timesteps=training_steps, callback=eval_callback)

# Alarm when training is complete
duration = 0.5  # seconds
freq = 880  # Hz
os.system("play -nq -t alsa synth {} sine {}".format(duration, freq))
