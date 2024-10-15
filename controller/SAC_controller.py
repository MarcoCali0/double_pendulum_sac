from stable_baselines3 import SAC

from controller.abstract_controller import AbstractController


class SACController(AbstractController):
    def __init__(self, model_path, dynamics_func, dt):
        super().__init__()

        self.model = SAC.load(model_path)
        self.dynamics_func = dynamics_func
        self.dt = dt

    def get_control_output_(self, x, t=None):
        obs = self.dynamics_func.normalize_state(x)
        action = self.model.predict(obs, deterministic=True)
        u = self.dynamics_func.unscale_action(action)
        return u
