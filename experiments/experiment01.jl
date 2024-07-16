using NACE
env = NACE.gym.make("MiniGrid-LavaCrossingS11N5-v0", render_mode="human")
obs, info = env.reset()

run_example_random(env)
