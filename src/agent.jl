
mutable struct Agent
    state::Any
    policy::Function
end

function (agent::Agent)(obs)
    action = agent.policy(agent.state, obs)
    action
end

"""
    NaceAgent(state, policy, perceptor, effector)

Non-Axiomatic Causal Explorer agent

Holds the top-level structure of the agent.
This is what you need to instantiate in order to run NACE.

# Arguments

  - `state` :: NaceState: Current state of the agent.
  - `policy` :: Function: Policy function that generates an action based on the current state.
  - `perceptor` :: Function: Perceptor function that generates perceived external state based on the
    received environment observation.
  - `effector` :: Function: Effector function that takes an action as input and returns data usable
    for executing the action via the environment's API.
"""
mutable struct NaceAgent
    state::NaceState
    policy::Function
    perceptor::Function
    effector::Function
end
