using NACE
env = NACE.gym.make("MiniGrid-LavaCrossingS11N5-v0", render_mode="human")

"""
    run_example(env, steps::Int)

Run an example on an environment.
"""
function run_example(env, steps::Int)
    obs, info = env.reset()
    agent = NaceAgent(init_state(), nace_policy, nace_perceptor, nace_effector)
    
    # Output initial state information
    @info "Starting experiment" environment="MiniGrid-LavaCrossingS11N5-v0" steps=steps
    
    # Run for the specified number of steps
    for step ∈ 1:steps
        action = agent(obs)
        @info "Step $step" action = IDX_TO_ACTION[action]
        obs, info = env.step(action)
        @debug "Step info" info = info
        
        # Periodically report on rules learned
        if step % 10 == 0
            num_rules = length(agent.state.rules)
            @info "Agent progress" step=step rules_count=num_rules
        end
    end
    
    # Write final rules to file for analysis
    rules_file = "output-rules-ex02.txt"
    write_rules_to_file(agent.state.rules, rules_file)
    @info "Experiment complete" total_steps=steps total_rules=length(agent.state.rules) rules_file=rules_file
end

run_example(env, 100)
