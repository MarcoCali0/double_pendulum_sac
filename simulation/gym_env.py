import gymnasium as gym
import numpy as np
from gym import logger
from gym.error import DependencyNotInstalled


class DoublePendulumEnv(gym.Env):
    def __init__(
        self,
        dynamics_func,
        reset_func,
        obs_space=gym.spaces.Box(
            np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0])
        ),
        mode="human",
        act_space=gym.spaces.Box(np.array([-1.0, -1.0]), np.array([1.0, 1.0])),
        max_episode_steps=1000,
        terminates=True,
        render_fps=60,
    ):
        self.dynamics_func = dynamics_func
        self.reset_func = reset_func
        self.observation_space = obs_space
        self.action_space = act_space
        self.max_episode_steps = max_episode_steps

        self.max_V = self.dynamics_func.simulator.plant.potential_energy(
            self.dynamics_func.unscale_state(np.array([0.0, -1.0, 0.0, 0.0]))
        )

        self.previous_action = 0

        self.terminates = terminates
        # For rendering
        self.mode = mode
        self.render_fps = render_fps
        self.SCREEN_DIM = 400
        self.screen = None
        self.clock = None
        self.isopen = True

        self.observation = self.reset_func()
        self.step_counter = 0
        self.stabilisation_mode = False
        self.y = [0.0, 0.0]

        l1 = self.dynamics_func.simulator.plant.l[0]
        l2 = self.dynamics_func.simulator.plant.l[1]
        self.max_height = l1 + l2

        if self.dynamics_func.robot == "acrobot":
            self.control_line = 0.9 * self.max_height
        elif self.dynamics_func.robot == "pendubot":
            self.control_line = 0.7 * self.max_height

    # Update the y coordinate of the first joint and the end effector
    def update_y(self):
        theta1, theta2, _, _ = self.dynamics_func.unscale_state(self.observation)

        link_end_points = self.dynamics_func.simulator.plant.forward_kinematics(
            [theta1, theta2]
        )
        self.y[0] = link_end_points[0][1]
        self.y[1] = link_end_points[1][1]

    def gravitational_reward(self):
        x = self.dynamics_func.unscale_state(self.observation)
        V = self.dynamics_func.simulator.plant.potential_energy(x)
        return V

    def V(self):
        return self.gravitational_reward()

    def kinetic_reward(self):
        x = self.dynamics_func.unscale_state(self.observation)
        T = self.dynamics_func.simulator.plant.kinetic_energy(x)
        return T

    def T(self):
        return self.kinetic_reward()

    def reward_func(self, terminated, action):
        _, theta2, omega1, omega2 = self.dynamics_func.unscale_state(self.observation)
        costheta2 = np.cos(theta2)

        a = action[0]
        delta_action = np.abs(a - self.previous_action)
        lambda_delta = 0.15
        lambda_action = 0.08
        lambda_velocities = 0.0007
        if not terminated:
            if self.dynamics_func.robot == "acrobot":
                if self.stabilisation_mode:
                    reward = (
                        self.V()
                        + 2 * (1 + costheta2) ** 2
                        - self.T()
                        - 5 * lambda_action * np.square(a)
                        - 3 * lambda_delta * delta_action
                    )
                else:
                    reward = (
                        (1 - np.abs(a)) * self.V()
                        - lambda_action * np.square(a)
                        - 2 * lambda_velocities * (omega1**2 + omega2**2)
                        - 3 * lambda_delta * delta_action
                    )

            elif self.dynamics_func.robot == "pendubot":
                if self.stabilisation_mode:
                    reward = (
                        self.V()
                        + 2 * (1 + costheta2) ** 2
                        - self.T()
                        - 5 * lambda_action * np.square(a)
                        - 3 * lambda_delta * delta_action
                    )
                else:
                    reward = (
                        self.V()
                        - lambda_action * np.square(a)
                        - 2 * lambda_velocities * (omega1**2 + omega2**2)
                        - 3 * lambda_delta * delta_action
                    )
        else:
            reward = -1.0
        return reward

    def terminated_func(self):
        if self.terminates:
            # Checks if we're in stabilisation mode and the ee has fallen below the control line
            if self.stabilisation_mode and self.y[1] < self.control_line:
                return True
        return False

    def step(self, action):
        self.observation = self.dynamics_func(self.observation, action)

        self.update_y()

        if self.y[1] >= self.control_line:
            self.stabilisation_mode = True

        terminated = self.terminated_func()
        reward = self.reward_func(terminated, action)

        truncated = False
        self.step_counter += 1
        if self.step_counter >= self.max_episode_steps:
            truncated = True
            self.step_counter = 0

        self.previous_action = action[0]
        return self.observation, reward, terminated, truncated, {}

    def reset(self, seed=0, options=None):
        super().reset(seed=seed)

        self.previous_action = 0
        self.observation = self.reset_func()
        self.step_counter = 0
        self.stabilisation_mode = False
        return self.observation, {}

    def render(self):
        if self.mode is None:
            logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the mode at initialization, "
                f'e.g. gym("{self.spec.id}", mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.SCREEN_DIM, self.SCREEN_DIM)
                )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.SCREEN_DIM, self.SCREEN_DIM))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        surf = pygame.Surface((self.SCREEN_DIM, self.SCREEN_DIM))
        surf.fill((255, 255, 255))

        l1 = self.dynamics_func.simulator.plant.l[0]
        l2 = self.dynamics_func.simulator.plant.l[1]

        bound = l1 + l2 + 0.2  # 2.2 for default
        scale = self.SCREEN_DIM / (bound * 2)
        offset = self.SCREEN_DIM / 2

        # s = self.scale_state()
        s = self.dynamics_func.unscale_state(self.observation)

        if s is None:
            return None

        p1 = [
            -l1 * np.cos(s[0]) * scale,
            l1 * np.sin(s[0]) * scale,
        ]

        p2 = [
            p1[0] - l2 * np.cos(s[0] + s[1]) * scale,
            p1[1] + l2 * np.sin(s[0] + s[1]) * scale,
        ]

        xys = np.array([[0, 0], p1, p2])[:, ::-1]
        thetas = [s[0] - np.pi / 2, s[0] + s[1] - np.pi / 2]
        link_lengths = [l1 * scale, l2 * scale]

        for (x, y), th, llen in zip(xys, thetas, link_lengths):
            x = x + offset
            y = y + offset
            l, r, t, b = 0, llen, 0.1 * scale, -0.1 * scale
            t, b = 0.02 * scale, -0.02 * scale

            coords = [(l, b), (l, t), (r, t), (r, b)]
            transformed_coords = []
            for coord in coords:
                coord = pygame.math.Vector2(coord).rotate_rad(th)
                coord = (coord[0] + x, coord[1] + y)
                transformed_coords.append(coord)
            gfxdraw.aapolygon(surf, transformed_coords, (0, 204, 204))
            gfxdraw.filled_polygon(surf, transformed_coords, (0, 204, 204))

            gfxdraw.filled_circle(
                surf, int(x), int(y), int(0.03 * scale), (204, 204, 0)
            )

        surf = pygame.transform.flip(surf, False, True)
        self.screen.blit(surf, (0, 0))

        # Display angle information
        pygame.font.init()
        font = pygame.font.SysFont("Comic Sans MS", 30)
        text_surface = font.render(
            f"Angle 1: {int(s[0] * 180/ np.pi) } | Angle 2: {int(s[1] * 180/np.pi)}",
            False,
            (0, 0, 0),
        )
        self.screen.blit(text_surface, (0, 0))

        if self.mode == "human":
            pygame.event.pump()
            self.clock.tick(self.render_fps)
            pygame.display.flip()

        elif self.mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


class double_pendulum_dynamics_func:
    def __init__(
        self,
        simulator,
        dt=0.01,
        integrator="runge_kutta",
        robot="acrobot",
        state_representation=2,
        max_velocity=20.0,
        torque_limit=[10.0, 10.0],
    ):
        self.simulator = simulator
        self.dt = dt
        self.integrator = integrator
        self.robot = robot
        self.state_representation = state_representation
        self.max_velocity = max_velocity
        self.torque_limit = torque_limit

    def __call__(self, state, action):
        x = self.unscale_state(state)
        u = self.unscale_action(action)
        xn = self.integration(x, u)
        obs = self.normalize_state(xn)
        return np.array(obs, dtype=np.float32)

    def integration(self, x, u):
        if self.integrator == "runge_kutta":
            next_state = np.add(
                x,
                self.dt * self.simulator.runge_integrator(x, self.dt, 0.0, u),
                casting="unsafe",
            )
        elif self.integrator == "euler":
            next_state = np.add(
                x,
                self.dt * self.simulator.euler_integrator(x, self.dt, 0.0, u),
                casting="unsafe",
            )
        elif self.integrator == "odeint":
            next_state = np.add(
                x,
                self.dt * self.simulator.odeint_integrator(x, self.dt, 0.0, u),
                casting="unsafe",
            )
        else:
            print("Invalid Integrator")
        return next_state

    def unscale_action(self, action):
        """
        scale the action
        [-1, 1] -> [-limit, +limit]
        """
        if self.robot == "double_pendulum":
            a = [
                float(self.torque_limit[0] * action[0]),
                float(self.torque_limit[1] * action[1]),
            ]
        elif self.robot == "pendubot":
            a = np.array([float(self.torque_limit[0] * action[0]), 0.0])
        elif self.robot == "acrobot":
            a = np.array([0.0, float(self.torque_limit[1] * action[0])])
        return a

    def unscale_state(self, observation):
        """
        scale the state
        [-1, 1] -> [-limit, +limit]
        """
        if self.state_representation == 2:
            x = np.array(
                [
                    observation[0] * np.pi + np.pi,
                    observation[1] * np.pi + np.pi,
                    observation[2] * self.max_velocity,
                    observation[3] * self.max_velocity,
                ]
            )
        elif self.state_representation == 3:
            x = np.array(
                [
                    np.arctan2(observation[0], observation[1]),
                    np.arctan2(observation[2], observation[3]),
                    observation[4] * self.max_velocity,
                    observation[5] * self.max_velocity,
                ]
            )
        return x

    def normalize_state(self, state):
        """
        rescale state:
        [-limit, limit] -> [-1, 1]
        """
        if self.state_representation == 2:
            observation = np.array(
                [
                    (state[0] % (2 * np.pi) - np.pi) / np.pi,
                    (state[1] % (2 * np.pi) - np.pi) / np.pi,
                    np.clip(state[2], -self.max_velocity, self.max_velocity)
                    / self.max_velocity,
                    np.clip(state[3], -self.max_velocity, self.max_velocity)
                    / self.max_velocity,
                ]
            )
        elif self.state_representation == 3:
            observation = np.array(
                [
                    np.cos(state[0]),
                    np.sin(state[0]),
                    np.cos(state[1]),
                    np.sin(state[1]),
                    np.clip(state[2], -self.max_velocity, self.max_velocity)
                    / self.max_velocity,
                    np.clip(state[3], -self.max_velocity, self.max_velocity)
                    / self.max_velocity,
                ]
            )

        return observation
