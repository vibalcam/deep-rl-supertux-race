import gym
from utils import dotdict

from baseline.aimPointController import AimPointController
from agents.abstractAgent import AbstractAgent

from environments.pytux import PyTux
import numpy as np

tracks = [
    "lighthouse",
    "hacienda",
    "snowtuxpeak",
    "zengarden",
    "cornfield_crossing",
    "scotland",
]
options = PyTux.default_params.copy()
options.update(dict(
    track=tracks[0],
    ai=None,
    render_every=1,
    n_karts=1,
    n_laps=1,
))
env = gym.make('PyTux-v0', options=options)


def get_trajectories(path:str="data", max_noise=(0.1,5)):
    options.render = False

    # for each track in the list
    for tr in tracks:
        p = f"{path}/{tr}/"
        options.track = tr

        # Get runs for AI
        for k in range(3):
            options.ai = k
            AbstractAgent(env, options).save_run(
                path=p+f"ai-{k}", 
                n_runs=10,
                save_every_steps=5,
            )
        
        # Get runs for baseline aim-controller
        for k in range(30):
            options.ai = None
            AimPointController(env, options).save_run(
                path=p+f"aim-{k}", 
                n_runs=1,
                save_every_steps=5,
                noise=np.random.rand(2) * max_noise,
            )

    env.close()




get_trajectories("data/1")