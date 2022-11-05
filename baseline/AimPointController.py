import numpy as np

from agents.AbstractAgent import AbstractAgent
from baseline.planner import load_model
from environments.pytux import VELOCITY, IMAGE, PyTux


class AimPointController(AbstractAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aim_planner = load_model().eval()

    def act(self, state: PyTux.State):
        super().act(state)
        img = self._to_torch(state[IMAGE])
        vel = np.linalg.norm(state[VELOCITY])
        aim_point = self._pred_aim_poing(img)
        return self._control(aim_point, vel)

    def _pred_aim_poing(self, img):
        return self.aim_planner(img[None]).squeeze(0).cpu().detach().numpy()

    def _control(self, aim_point, current_vel) -> PyTux.Action:
        action = PyTux.Action()
        # *Controller
        # - cornfield_crossing[64.6 s]
        # - hacienda[53.1 s]
        # - lighthouse[43.3 s]
        # - scotland[56.5 s]
        # - snowtuxpeak[45.9 s]
        # - zengarden[42.4 s]
        abs_x = np.abs(aim_point[0])
        action.acceleration = (1 if current_vel < 35 and abs_x < 0.4 else 0) if current_vel > 8 else 1
        action.steer = np.tanh(current_vel * aim_point[0])
        # action.steer = np.tanh(2 * current_vel * aim_point[0])
        action.drift = abs_x > 0.2 or (abs_x > 0.15 and current_vel > 15)
        action.brake = current_vel > 20 and abs_x > 0.4
        action.nitro = abs_x < 0.5

        return action
