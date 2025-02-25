using Test

using NACE

@testset "NACE.jl" begin
    @testset "unit:base" begin
        # Create a test state with some values
        test_state = init_state()
        # Add some values to test state
        test_state = NaceState(
            test_state.t,
            test_state.focus,
            test_state.rules,
            [0, 0, 0],  # Add some values
            test_state.context,
        )

        # Create a rule with the new structure
        # Note: In our coordinate system, x is the column (horizontal) and y is the row (vertical)
        # Cell(x, y, item)
        rule = make_rule(
            test_state,
            :test,
            Cell(3, 3, "red"),  # Consequence cell at (3,3)
            Cell(3, 2, "blue"),  # Precondition cell 1 - one row up (x+0, y-1)
            Cell(2, 3, "green"), # Precondition cell 2 - one column left (x-1, y+0)
            "Move forward",
        )

        # Test the precondition contains the correct relative coordinates
        @test length(rule.precondition.conditions) == 2

        # Check if we have a condition with x_offset = 0, y_offset = -1, and value = "blue"
        # This represents the cell one row above the consequence cell
        has_blue_condition = false
        for c ∈ rule.precondition.conditions
            if c.x_offset == 0 && c.y_offset == -1 && c.value == "blue"
                has_blue_condition = true
                break
            end
        end
        @test has_blue_condition

        # Check if we have a condition with x_offset = -1, y_offset = 0, and value = "green"
        # This represents the cell one column to the left of the consequence cell
        has_green_condition = false
        for c ∈ rule.precondition.conditions
            if c.x_offset == -1 && c.y_offset == 0 && c.value == "green"
                has_green_condition = true
                break
            end
        end
        @test has_green_condition

        # Test the consequence has the correct value
        @test rule.consequence.cell_value == "red"
        @test rule.consequence.target_cell.x == 3
        @test rule.consequence.target_cell.y == 3

        # Test the rule has the correct action
        @test rule.precondition.action == "Move forward"

        # Test initial evidence and score values
        @test rule.evidence_pos == 0.0f0
        @test rule.evidence_neg == 0.0f0
        @test rule.score == 0.0f0

        # Test string representation (format may vary slightly but should contain key elements)
        rule_str = string(rule)
        @test occursin("Action: Move forward", rule_str)
        @test occursin("Precondition:", rule_str)
        @test occursin("blue at", rule_str)
        @test occursin("green at", rule_str)
        @test occursin("Consequence: red", rule_str)

        # Test rule validity using NACE.is_valid_rule to be explicit
        @test NACE.is_valid_rule(rule, Set{Rule}())

        # Test truth expectation with no evidence
        @test NACE.truthexp(rule) ≈ 0.5f0

        # Test truth expectation with some evidence
        rule.evidence_pos = 2.0f0
        rule.evidence_neg = 1.0f0
        @test NACE.truthexp(rule) > 0.5f0

        # Test rule matching
        # Create two rules with same conditions but different consequences
        rule2 = make_rule(
            test_state,
            :test,
            Cell(3, 3, "yellow"),  # Different consequence
            Cell(3, 2, "blue"),    # Same precondition
            Cell(2, 3, "green"),   # Same precondition
            "Move forward",         # Same action
        )

        # Test conflict detection with explicit NACE reference
        @test NACE.conflicting_rule_exists(rule, Set{Rule}([rule2]))
    end
    @testset "integration:env" begin
        env = NACE.gym.make("MiniGrid-LavaCrossingS11N5-v0", render_mode="human")
        obs, info = env.reset()

        @test run_example_random(env)
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
            for step ∈ 1:5
                action = agent(obs)
                @info "Step $step" action = IDX_TO_ACTION[action]
                obs, info = env.step(action)
                @debug "Step info" info = info
            end

            # Write final rules to file
            write_rules_to_file(agent.state.rules, "output-rules-test.txt")
            true
        end
        @test run_example(env)
    end
end
