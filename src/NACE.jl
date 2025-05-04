module NACE
export make_random_policy,
    run_example_random,
    NaceAgent,
    nace_effector,
    nace_perceptor,
    nace_policy,
    IDX_TO_ACTION,
    ACTION_TO_IDX,
    IDX_TO_OBJECT,
    OBJECT_TO_IDX,
    # Additional exports for the test suite
    is_valid_rule,
    conflicting_rule_exists,
    Cell,
    Rule,
    ValueTuple,
    RelativeCondition,
    Precondition,
    Consequence,
    make_rule,
    init_state,
    truthexp

using DataStructures
using PyCall

function __init__()
    global gym = pyimport("gymnasium")
    global miniwrap = pyimport("minigrid.wrappers")
end

include("base.jl")
include("env.jl")
include("agent.jl")

# example of running in the REPL
# env = NACE.gym.make("MiniGrid-LavaCrossingS11N5-v0", render_mode="human");
# obs, info = env.reset();

"""
    make_random_policy(env)

Return a function that generates a random policy for the given environment.
"""
function make_random_policy(env)
    n = convert(Int, env.action_space.n)

    function random_policy(state::Any, observation)
        return rand(0:n-1)
    end

    random_policy
end

"""
    run_example_random(env)

@deprecated Run a random policy on an environment.
"""
function run_example_random(env)
    obs, info = env.reset()
    agent = Agent{Any}(nothing, make_random_policy(env))
    for _ ∈ 1:10
        action = agent(obs)
        println("Action: $(IDX_TO_ACTION[action])")
        obs, info = env.step(action)
        println("Info: $info")
    end
    true
end

"""
    (agent::NaceAgent)(obs)

Run a step

Run the complete pipeline from perceiving from the environment observation to
determining the next action to take.
Returns the chosen action, does not have side-effects except for updating the agent's state.
"""
function (agent::NaceAgent)(obs)
    percept_state = agent.perceptor(obs)
    # Get values from the board matrix
    values = if !isempty(percept_state.board)
        map(
            cell -> cell.item == "goal" ? 1 : (cell.item == "lava" ? -1 : 0),
            vec(percept_state.board),
        )
    else
        Int[]
    end
    per_ext_ante = if isempty(agent.state.memory.episodic_antecedant)
        percept_state
    else
        agent.state.memory.episodic_antecedant
    end
    new_memory = Memory(percept_state, per_ext_ante, agent.state.memory.act_ante)
    agent.state =
        NaceState(agent.state.t, agent.state.focus, agent.state.rules, values, new_memory)
    agent.state = cycle(agent.state)
    agent.effector(agent.policy(agent.state))
end

"""
    nace_perceptor(obs)

Run the perceptor

Run the perceptor -- the function that consumes an environment observation
data structure and returns a representation usable within the agent's internal
logic.
"""
function nace_perceptor(obs)
    # Create the board representation
    objects = map(i -> IDX_TO_OBJECT[i], obs["image"][:, :, 1])
    board = reshape(
        [Cell(idx[2], idx[1], objects[idx]) for idx ∈ CartesianIndices(objects)],
        size(objects),
    )
    values = map(obj -> obj == "goal" ? 1 : (obj == "lava" ? -1 : 0), vec(objects))

    # Create EpisodicMemory with board as the main field
    return EpisodicMemory(board, Direction(obs["direction"]))
end

"""
    nace_policy(state)

Run the policy
"""
function nace_policy(state)
    state.memory.act_ante
end

"""
    nace_effector(action)

Run the effector

Run the effector -- the function that takes an action as input and returns its representation
usable with the environment's API.
"""
function nace_effector(action)
    ACTION_TO_IDX[action]
end

end # module
