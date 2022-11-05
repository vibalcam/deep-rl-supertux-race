from gym.envs.registration import register

register(
    id='PyTux-v0',
    entry_point='environments.pytux:PyTux',
    # max_episode_steps=2000,
)