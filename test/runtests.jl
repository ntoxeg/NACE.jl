using Test

using NACE

@testset "NACE.jl" begin
    @testset "unit:base" begin
        rule = make_rule(
            init_state(),
            :test,
            Cell(1, 1, "red"),
            Cell(2, 2, "blue"),
            Cell(1, 1, "red"),
            "move",
        )
        @test rule.precondition.cell1 == Cell(1, 1, "red")
        @test rule.precondition.cell2 == Cell(2, 2, "blue")
        @test rule.consequence.cell == Cell(1, 1, "red")

        expected = """Rule[
Precondition: if red and blue then red,
Consequence: red,
Score: 0.0
]"""
        @test string(rule) == expected
    end
    @testset "integration:agent" begin
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
            true
        end
        @test run_example(env)
    end
end
