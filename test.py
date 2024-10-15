import sys

import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.utils import set_random_seed
from tabulate import tabulate

import constants
from controller.SAC_controller import SACController
from model.model_parameters import model_parameters
from model.symbolic_plant import SymbolicDoublePendulum
from simulation.gym_env import DoublePendulumEnv, double_pendulum_dynamics_func
from simulation.simulation import Simulator
from utils.metrics import get_swingup_time
from utils.plotting import energy_plot, plot, plot_timeseries, rewards_plot
from utils.reset_functions import *

# MODEL SETTING
acrobot_flag = "--acrobot" in sys.argv
pendubot_flag = "--pendubot" in sys.argv

if (acrobot_flag and pendubot_flag) or not (acrobot_flag or pendubot_flag):
    print("You can only input either pendubot or acrobot")
    exit(1)


robot = "acrobot" if acrobot_flag else "pendubot"
print(f"---- Running SAC for {robot} ----")

# Set random seed to zero for reproducibility
set_random_seed(0)

max_velocity = constants.max_velocity
max_torque = constants.max_torque

torque_limit = [max_torque, 0.0] if robot == "pendubot" else [0.0, max_torque]
active_act = 0 if robot == "pendubot" else 1
model_par_path = f"parameters/common_parameters.yml"
# model_path = f"log_data/SAC_training/evaluations/acrobot_model_14.zip"
model_path = constants.model_path

# Model parameters
mpar = model_parameters(filepath=model_par_path)
print("Loading model parameters...")

# Simulation parameters
dt = 0.01
max_steps = 1_000
t_final = max_steps * dt
print(f"Loading simulation settings: dt = {dt}s, T = {t_final}s")


# learning environment parameters
state_representation = 2
obs_space = gym.spaces.Box(
    np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0])
)
act_space = gym.spaces.Box(np.array([-1]), np.array([1]))

integrator = "runge_kutta"
plant = SymbolicDoublePendulum(model_pars=mpar)
simulator = Simulator(plant=plant, integrator_name=integrator)

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

### Saving settings
save_video = "--save_video" in sys.argv
render_mode = "rgb_array" if save_video else "human"

print("Creating environment...")
# initialize vectorized environment
env = DoublePendulumEnv(
    dynamics_func=dynamics_func,
    reset_func=zero_reset_func,
    obs_space=obs_space,
    act_space=act_space,
    max_episode_steps=max_steps,
    mode=render_mode,
    terminates=False,
)

print("Loading trained model...")
controller = SACController(model_path=model_path, dynamics_func=dynamics_func, dt=dt)
controller.init()
print("SAC controller initialised")

obs, _ = env.reset()
done, truncated = False, False
score, i = 0, 0
torques, thetas1, thetas2, omegas1, omegas2 = [], [], [], [], []
kinetic_energy_vals, potential_energy_vals = [], []
rewards = []
T = []
X = []
t = 0

env_simulation = False
if env_simulation:
    if save_video:
        video_path = f"{robot}.mp4"
        print(f"Saving video as {video_path}")
        fps = 500
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, fourcc, fps, (400, 400))

    while not (truncated or done):
        if save_video:
            frame = env.render()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        else:
            env.render()

        T.append(t)
        x = dynamics_func.unscale_state(obs)
        u = controller.get_control_output(x)
        obs, reward, done, truncated, _ = env.step(
            (np.array(u[active_act] / max_torque, dtype=np.float32), None)
        )
        score += reward

        kinetic_energy_vals.append(plant.kinetic_energy(x))
        potential_energy_vals.append(plant.potential_energy(x))

        # Save data for plotting
        theta1, theta2, omega1, omega2 = x
        thetas1.append(theta1)
        thetas2.append(theta2)
        omegas1.append(omega1)
        omegas2.append(omega2)
        rewards.append(reward)

        tau = np.round(u[active_act], 2)
        torques.append(tau)
        X.append(list(x))

        omega = np.round([omega1, omega2], 2)
        r = np.round(reward, 2)
        data = [
            ["Step", "Torque τ [Nm]", "Angular Velocity ω [rad/s]", "Reward"],
            [i, tau, omega, r],
        ]

        # Display data in tabular format
        print(tabulate(data, headers="firstrow"))
        print("")

        t = i * dt
        i += 1

    print(f"Final Return: {int(score-1)}")

    if save_video:
        out.release()

    env.close()

    plot(thetas1, thetas2, omegas1, omegas2, torques, robot, dt, t_final)
    rewards_plot(rewards, robot, env.max_V + 8, dt, t_final)
    energy_plot(kinetic_energy_vals, potential_energy_vals, robot, dt, t_final)

    print(
        "Swingup time: "
        + str(
            get_swingup_time(T, np.array(X), mpar=mpar, has_to_stay=True, height=0.95)
        )
        + "s"
    )

else:
    T, X, U = simulator.simulate_and_animate(
        t0=0.0,
        x0=[0.0, 0.0, 0.0, 0.0],
        tf=t_final,
        dt=dt,
        controller=controller,
        integrator=integrator,
        save_video=save_video,
    )

    # plot time series
    plot_timeseries(
        T,
        X,
        U,
        X_meas=simulator.meas_x_values,
        pos_y_lines=[np.pi],
        tau_y_lines=[-torque_limit[active_act], torque_limit[active_act]],
    )
