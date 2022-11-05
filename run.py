import gym

from baseline.AimPointController import AimPointController

env = gym.make('PyTux-v0', options=dict(
    track='lighthouse',
    ai=False,
    render=True,
    n_karts=1,
    n_laps=1,
))

aim_controller = AimPointController(env).eval()
# ai_baseline =

# obs, _ = env.reset()
#
# count = 0
# # 437 for ai
# agent = AimPointController()
# for _ in range(2000):
#     # stateo = agent._to_torch(obs)
#     # obs, reward, done, _, _ = env.step(env.action_space.sample())
#     obs, reward, done, _, _ = env.step(agent.act(obs))
#     env.render()
#     count += 1
#     print(count)
#
#     if done:
#         break
#
# env.close()