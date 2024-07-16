using NACE
env = NACE.gym.make("MiniGrid-LavaCrossingS11N5-v0", render_mode="human")
obs, info = env.reset()

# TODO: move code from NACE.jl here.
run_example(env)
