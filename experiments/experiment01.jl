using NACE
env = NACE.gym.make("MiniGrid-LavaCrossingS11N5-v0", render_mode="human")
obs, info = env.reset()

runthis_random(env)
