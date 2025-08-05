using NACE
env = NACE.gym.make("MiniGrid-LavaCrossingS11N5-v0", render_mode="human")

"""
    run_random_example(env)

Run a random policy on an environment.
"""
function run_random_example(env)
    obs, info = env.reset()
    agent = NACE.Agent{Any}(nothing, make_random_policy(env))
    for _ ∈ 1:100
        action = agent(obs)
        println("Action: $(IDX_TO_ACTION[action])")
        obs, info = env.step(action)
    end
end

run_random_example(env)
