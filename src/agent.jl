
mutable struct Agent
    state::Any
    policy::Function
end

function (agent::Agent)(obs)
    action = agent.policy(agent.state, obs)
    action
end


