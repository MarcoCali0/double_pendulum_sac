import constants
from controller.SAC_controller import SACController
from model.model_parameters import model_parameters
from model.symbolic_plant import SymbolicDoublePendulum
from simulation.gym_env import double_pendulum_dynamics_func
from simulation.simulation import Simulator

name = "sac"
leaderboard_config = {
    "csv_path": name + "/sim_swingup.csv",
    "name": name,
    "simple_name": "sac",
    "short_description": "SAC for both swingup and stabilisation",
    "readme_path": f"readmes/{name}.md",
    "username": "MarcoCali0",
}


robot = constants.robot
print("Loading trained model...")

max_velocity = constants.max_velocity
max_torque = constants.max_torque

torque_limit = [max_torque, 0.0] if robot == "pendubot" else [0.0, max_torque]
active_act = 0 if robot == "pendubot" else 1
model_par_path = f"parameters/common_parameters.yml"
model_path = constants.model_path

# Model parameters
mpar = model_parameters(filepath=model_par_path)
mpar.set_torque_limit(torque_limit)

print("Loading model parameters...")

# Simulation parameters
dt = 0.01

# learning environment parameters
state_representation = 2

integrator = "runge_kutta"
plant = SymbolicDoublePendulum(model_pars=mpar)
simulator = Simulator(plant=plant, integrator_name=integrator)

dynamics_func = double_pendulum_dynamics_func(
    simulator=simulator,
    dt=dt,
    integrator=integrator,
    robot=robot,
    state_representation=state_representation,
    max_velocity=max_velocity,
    torque_limit=torque_limit,
)

controller = SACController(model_path=model_path, dynamics_func=dynamics_func, dt=dt)
controller.init()
