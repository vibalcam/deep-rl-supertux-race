import numpy as np

from agents.abstractAgent import AbstractAgent
from baseline.planner import load_model
from environments.pytux import VELOCITY, IMAGE, PyTux


class AimPointController(AbstractAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aim_planner = load_model().eval()

    def act(self, state: PyTux.State, noise=None):
        '''The function takes in a state and a noise value, and returns an action. 
        
        The action is calculated given an aim point and a velocity. 
        The aim point is predicted by a CNN from the image.
        
        Parameters
        ----------
        state : PyTux.State
            PyTux.State
        noise
            a tuple of two values, the first value is the noise added to the aim point, the second value is the noise added to the velocity.
        
        Returns
        -------
            The action obtained given the aim point and velocity
        
        '''
        super().act(state)

        if noise is None:
            # first value for aim point, second value for velocity
            noise = (0,0)

        img = self._to_torch(state[IMAGE])
        # calculate velocity and add noise
        vel = np.linalg.norm(state[VELOCITY]) + np.random.randn() * noise[1]
        # calculate aim point
        aim_point = self._pred_aim_poing(img)
        # add noise to aim point
        aim_point += np.random.randn(*aim_point.shape) * noise[0]
        # return action obtained given the aim point and velocity
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
