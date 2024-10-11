using NACE
env = NACE.gym.make("MiniGrid-LavaCrossingS11N5-v0", render_mode="human")

"""
Actions available in minigrid environments
"""
IDX_TO_ACTION = Dict(
    0 => "Turn left",
    1 => "Turn right",
    2 => "Move forward",
    3 => "Unused",
    4 => "Unused",
    5 => "Unused",
    6 => "Unused",
)
ACTION_TO_IDX = Dict(value => key for (key, value) ∈ IDX_TO_ACTION)

"""
    run_example(env)

Run an example on an environment.
"""
function run_example(env)
    obs, info = env.reset()
    agent = NaceAgent(
        init_state(),
        nace_policy,
        nace_perceptor,
        nace_effector,
    )
    for _ ∈ 1:10
        action = agent(obs)
        println("Action: $(IDX_TO_ACTION[action])")
        obs, info = env.step(action)
        println("Current rules: $(agent.state.rules)")
    end
end

run_example(env)
