from typing import Optional, Tuple, Dict

import gym
import matplotlib.pyplot as plt
import numpy as np
import pystk

from utils import dotdict

VELOCITY = 'vel'
IMAGE = 'img'
ROTATION = "rot"
ACCELERATION = "acceleration"
BRAKE = "brake"
DRIFT = "drift"
STEER = "steer"


class PyTux(gym.Env):
    _singleton = None
    _t_reward = -1
    _rescue_timer = 30
    default_params = dotdict(
        track='lighthouse',
        ai=None,
        render_every=1,
        n_karts=1,
        n_laps=1,
        reverse=False,
        log_every=10,
    )

    class State(dotdict):
        def __init__(self, image, velocity, rotation):
            super().__init__()
            self[IMAGE] = image
            self[VELOCITY] = velocity
            self[ROTATION] = rotation

    class Action(dotdict):
        def __init__(self, acceleration=0, brake=0, drift=0, steer=0):
            super().__init__()
            self[ACCELERATION]=acceleration
            self[BRAKE]=brake
            self[DRIFT]=drift
            self[STEER]=steer

        def to_pystk(self):
            a = pystk.Action()
            a.acceleration = self[ACCELERATION]
            a.brake = self[BRAKE]
            a.drift = self[DRIFT]
            a.steer = self[STEER]
            return a

    def __init__(self, screen_width:int =128, screen_height:int =96, options:Dict = None):
        self.param = self.default_params.copy()
        if options is not None:
            self.param.update(options)

        # set up pytux environment
        assert PyTux._singleton is None, "Cannot create more than one environment"
        PyTux._singleton = self
        self.config = pystk.GraphicsConfig.hd()
        self.config.screen_width = screen_width
        self.config.screen_height = screen_height
        pystk.init(self.config)

        # contains information about the world
        self._state = pystk.WorldState()
        # contains information about the track
        self._track = pystk.Track()
        # current race object
        self.race = None
        # last timestep when the player i was rescued
        self._last_rescue = None
        # current timestep
        self.t = 0
        # number of times it has been rescued
        self.n_rescued = 0
        # current maximum distance completed
        self.max_distance = 0
        # laps left
        self._laps_left = 0
        self._ax = None
        self._fig = None

        """
        States space
        ------------
            img: image of the game
            vel: current kart´s velocity
            rot: current kart´s rotation
        """
        self.observation_space = gym.spaces.Dict(
            {
                IMAGE: gym.spaces.Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8),
                VELOCITY: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                ROTATION: gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
            }
        )
        """
        Action space
        ------------
            acceleration: acceleration value for the kart
            brake: bool, true if todo g
            drift: 
            steer: 
        """
        self.action_space = gym.spaces.Dict(
            {
                ACCELERATION: gym.spaces.Box(low=0, high=1, dtype=np.float32),
                BRAKE: gym.spaces.Discrete(2),
                DRIFT: gym.spaces.Discrete(2),
                STEER: gym.spaces.Box(low=-1, high=1, dtype=np.float32),
            }
        )

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> Tuple[State, dict]:
        super().reset(seed=seed)

        # parameters for the new race
        #param = self.param.copy()
        if options is not None:
            self.param.update(options)

        # reset the race

        # if self.race is not None and self.race.config.track == self.param['track']:
        #     self.race.restart()
        # else:
        if self.race is not None:
            self.race.stop()
            del self.race
        import os
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        config = pystk.RaceConfig(
            num_kart=self.param['n_karts'], 
            laps=self.param['n_laps'], 
            track=self.param['track'],
            difficulty=self.param.ai if self.param.ai is not None else 0,
            reverse=self.param.reverse,
        )
        if self.param['ai'] is None:
            config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL
        else:
            config.players[0].controller = pystk.PlayerConfig.Controller.AI_CONTROL

        print(f"Start with ai {self.param.ai}, {self.param.track}")

        self.race = pystk.Race(config)
        self.race.start()

        # reset variables
        self._last_rescue = np.zeros(self.param['n_karts'])
        self._laps_left = self.param['n_laps']
        self.t = 0
        self.n_rescued = 0
        # set up to render image
        if self.param['render_every'] > 0:
            self._fig, self._ax = plt.subplots(1, 1)

        # take a step to obtain initial observations
        self.race.step()

        return self._update_and_obs(), self._get_info()

    def _update_and_obs(self):
        self._state.update()
        self._track.update()

        # calculate new state
        image = np.array(self.race.render_data[0].image)
        kart = self._state.players[0].kart
        velocity = np.asarray(kart.velocity, dtype=np.float32)
        rotation = np.asarray(kart.rotation, dtype=np.float32)

        return PyTux.State(
            image= image,
            velocity= velocity,
            rotation= rotation,
        )

    def _calc_reward(self):
        reward = 0
        # reward due to time
        reward += self._t_reward
        # reward due to distance
        kart = self._state.players[0].kart
        if kart.overall_distance > self.max_distance:
            reward += (kart.overall_distance - self.max_distance)
            self.max_distance = kart.overall_distance
        # reward due to being near center of track
        # todo finish reward function

        return reward


    def _get_info(self):
        kart = self._state.players[0].kart
        return dotdict(
            perc_completed=float(kart.overall_distance) / self._track.length,
            laps_left=self._laps_left,
        )

    def step(self,action) -> Tuple[State, float, bool, bool, dict]:
        self.t += 1
        obs = self._update_and_obs()
        reward = self._calc_reward()

        # check whether the player 0 has finished the race
        terminated = False
        kart = self._state.players[0].kart
        if np.isclose(kart.overall_distance / self._track.length, 1.0, atol=2e-3):
            self._laps_left -= 1
            if self._laps_left == 0:
                terminated = True
                print(f"Track {self.param.track} finished in {self.t}")

        # if not terminated, take action
        if not terminated:
            if not isinstance(action, PyTux.Action):
                a = PyTux.Action()
                a.update(action)
                action = a
                
            a = action.to_pystk()
            # check whether we should rescue the kart
            if np.linalg.norm(obs[VELOCITY]) < 1.0 and self.t - self._last_rescue > self._rescue_timer:
                self._last_rescue = self.t
                a.rescue = True
                self.n_rescued+=1

            # take step
            self.race.step(a)

            if self.t % self.param.log_every == 0:
                print(f"Track {self.param.track} step {self.t}")

            if self.param['render_every'] > 0 and self.t % self.param['render_every'] == 0:
                self.render(reward)

        return obs, reward, terminated, False, self._get_info()

    def render(self, reward):
        self._ax.clear()
        self._ax.imshow(self.race.render_data[0].image)
        plt.title(f"Last reward: {reward}")
        plt.pause(1e-3)

    def close(self):
        if self.race is not None:
            self.race.stop()
            del self.race
        pystk.clean()

