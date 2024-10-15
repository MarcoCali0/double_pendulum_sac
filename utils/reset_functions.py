import numpy as np


# used for evaluation
def zero_reset_func():
    observation = [-1.0, -1.0, 0.0, 0.0]
    return observation


# used for training
def noisy_reset_func():
    scale = 0.05
    rand = np.random.rand(4) * scale
    rand[2:] -= scale / 2
    observation = [-1.0, -1.0, 0.0, 0.0] + rand
    return observation


def stabilisation_reset_func():
    observation = [0.0, -1.0, 0.0, 0.0]
    return observation


def noisy_stabilisation_reset_func():
    scale = 0.05
    rand = np.random.rand(4) * scale
    rand[2:] -= scale / 2  # velocity noise
    rand[0] -= scale / 2  # theta1 noise
    observation = [0.0, -1.0, 0.0, 0.0] + rand
    return observation
