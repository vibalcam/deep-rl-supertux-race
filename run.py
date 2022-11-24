import gym
from utils import dotdict, load_model, set_seed

from baseline.aimPointController import AimPointController
from agents.decision_transformer import MODEL_CLASS, TransformerController, AgentTransformerModel
from agents.cnn import KartCNN
from agents.abstractAgent import AbstractAgent

from environments.pytux import PyTux
import numpy as np
import pandas as pd
from collections import defaultdict

tracks = [
    # todo test other tracks
    "lighthouse",
    # "hacienda",
    # "snowtuxpeak",
    # "cornfield_crossing",
    # "zengarden",
    # "scotland",
]
options = PyTux.default_params.copy()
options.update(dict(
    track=tracks[0],
    ai=None,
    render_every=0,
    n_karts=1,
    n_laps=1,
    no_pause_render=True,
    # save_video='video.mp4',
    # save_imgs='tmp/imgs',
))
env = gym.make('PyTux-v0', screen_width = 128, screen_height = 96, options=options)


def evaluate(n_runs: int=10, save_videos: bool = True):
    base_seed = 1234
    stats = {}
    # options for video
    options_video = options.copy()
    options_video.render_every = 1
    options_video.no_pause_render = True
    # options for no video
    options_no_video = options.copy()
    options_no_video.render_every = 0

    for tr in tracks:
        track_stats = defaultdict(lambda: [])

        for r in range(n_runs):
            options_current = options_video if r==0 and save_videos else options_no_video
            options_current.track = tr

            # Game AI
            for k in range(3):
                name = f"ai_{k}"
                # set options
                options_current.save_video=f"evaluate/{tr}_{name}.mp4"
                options_current.seed=base_seed + r
                set_seed(base_seed + r)
                options_current.ai = k
                # run model and save results
                res = AbstractAgent(env, options_current).evaluate()
                track_stats[name].append(res)

            # NON AI MODELS
            options_current.ai = None

            # todo Aim point controller no drift fixed velocity
            # todo Aim point controller fixed velocity
            # todo Aim point controller noisy

            # Aim point controller
            name = f"baseline"
            # set options
            options_current.save_video=f"evaluate/{tr}_{name}.mp4"
            options_current.seed=base_seed + r
            set_seed(base_seed + r)
            # run model and save results
            res = AimPointController(env, options_current).evaluate()
            track_stats[name].append(res)

            # Decision transformer controller no drift fixed velocity 1
            l = [
                'decTransColor1_80',
                'decTransColor1_best',
            ]
            for k in l:
                name = f"trans_no_drift_{k}"
                # set options
                options_current.save_video=f"evaluate/{tr}_{name}.mp4"
                options_current.seed=base_seed + r
                set_seed(base_seed + r)
                # run model and save results
                res = TransformerController(
                    env, 
                    options=options_current,
                    target_reward=500,
                    model=load_model(f"./saved/trans/colorNoDrift/{k}", MODEL_CLASS)[0],
                    fixed_velocity=1,
                ).evaluate()
                track_stats[name].append(res)

            # Decision transformer controller fixed velocity 1
                # l = ['colorDrift_tmp2/decTransColor_drift1_best',
                #     'colorDrift_tmp2/decTransColor_drift1_159',
                #     'colorDrift_tmp3/decTransColor_drift1_best',
                #     'colorDrift_tmp2/decTransColor_drift1_299',
                #     'colorDrift_tmp3/decTransColor_drift1_299',
                #     'colorDrift_tmp/decTransColor_drift1_best',
                #     'colorDrift_tmp/decTransColor_drift1_99',
                #     'colorDrift/decTransColor_drift1_best',
                #     'colorDrift/decTransColor_drift1_139',
                # ]
            l = [
                # 1 Best and lowest variability
                'colorDrift_tmp2/decTransColor_drift1_best',
                # 2 Best, medium variability
                'colorDrift_tmp/decTransColor_drift1_best',
                # 3 Best, medium variability
                'colorDrift_tmp3/decTransColor_drift1_best',
                # 4 Best, high variability
                'colorDrift/decTransColor_drift1_139',
            ]
            for k,p in enumerate(l):
                name = f"trans_drift_{k}"
                # set options
                options_current.save_video=f"evaluate/{tr}_{name}.mp4"
                options_current.seed=base_seed + r
                set_seed(base_seed + r)
                # run model and save results
                res = TransformerController(
                    env, 
                    options=options_current,
                    target_reward=500,
                    allow_drift=True,
                    model=load_model(f"./saved/trans/{p}", MODEL_CLASS)[0],
                    fixed_velocity=1,
                ).evaluate()
                track_stats[name].append(res)

        # calculate mean stats
        temp_stats = {}
        for k1,v in track_stats.items():
            mean_d = {}
            for k2 in v[0].keys():
                mean_d[k2+"_mean"] = np.mean([i[k2] for i in v])
                mean_d[k2+"_std"] = np.std([i[k2] for i in v])
                mean_d[k2+"_max"] = np.max([i[k2] for i in v])
                mean_d[k2+"_min"] = np.min([i[k2] for i in v])
                mean_d[k2+"_dif"] = mean_d[k2+"_max"] - mean_d[k2+"_min"]
            temp_stats[k1] = mean_d
        
        # associate stats with track
        stats[tr] = pd.DataFrame.from_dict(temp_stats, orient='index')

    # concatenate dataframes from different tracks
    res = pd.concat(stats.values(), axis=1, keys=stats.keys())
    res.round(2).to_excel('evaluate.xlsx')
    return res


def get_trajectories(path:str="data", max_noise=(0.1,5)):
    options.render_every = 0
    max_noise = np.asarray(max_noise)

    # for each track in the list
    for tr in tracks:
        p = f"{path}/{tr}/"
        options.track = tr

        # # Get runs for AI
        # for k in range(3):
        #     options.ai = k
        #     AbstractAgent(env, options).save_run(
        #         path=p+f"ai-{k}", 
        #         n_runs=10,
        #         save_every_steps=5,
        #     )
        
        # Get runs for baseline aim-controller
        for dr in [True, False]:    # True disables drift, False enables it
            nm = 0
            # No noise
            for k in range(30):
                options.ai = None
                AimPointController(env, options=options, disable_drift=dr).save_run(
                    path=p+f"aim-dr{dr}-noise{max_noise * nm}-{k}", 
                    n_runs=1,
                    save_every_steps=1,
                    noise=np.random.rand(2) * max_noise * nm,
                )
            # With noise
            for nm in [0.5,1]:
                for k in range(60):
                    options.ai = None
                    AimPointController(env, options=options, disable_drift=dr).save_run(
                        path=p+f"aim-dr{dr}-noise{max_noise * nm}-{k}", 
                        n_runs=1,
                        save_every_steps=1,
                        noise=np.random.rand(2) * max_noise * nm,
                    )


# todo put get data no drift

# todo put get data with/without drift


import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--path', '-p', default="./dataNoDrift/1")
parser.add_argument('--trajectories', '-t', action='store_true', help="Save trajectories")
parser.add_argument('--evaluate', '-e', action='store_true', help="Evaluate controllers")

args = parser.parse_args()

if args.trajectories:
    # todo Dataset no drift: 
    # todo Dataset with/without drift:
    get_trajectories(args.path)
elif args.evaluate:
    evaluate()
else:
    # get_trajectories(args.path)
    evaluate()

env.close()