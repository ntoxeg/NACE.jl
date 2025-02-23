using Test

using NACE

@testset "NACE.jl" begin
    env = NACE.gym.make("MiniGrid-LavaCrossingS11N5-v0", render_mode="human")

    """
        run_example(env)

    Run an example on an environment.
    """
    function run_example(env)
        obs, info = env.reset()
        agent = NaceAgent(init_state(), nace_policy, nace_perceptor, nace_effector)
        for _ ∈ 1:10
            action = agent(obs)
            println("Action: $(NACE.IDX_TO_ACTION[action])")
            obs, info = env.step(action)
            println("Current rules: $(agent.state.rules)")
        end
    end

    run_example(env)
end
