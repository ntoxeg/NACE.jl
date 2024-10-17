module NACE
export make_random_policy,
    run_example_random, NaceAgent, init_state, nace_effector, nace_perceptor, nace_policy

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

    function random_policy(state, observation)
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
    agent = Agent(nothing, make_random_policy(env))
    for _ ∈ 1:10
        action = agent(obs)
        println("Action: $(IDX_TO_ACTION[action])")
        obs, info = env.step(action)
        println("Info: $info")
    end
end

"""
    init_state()

Create an empty state with time step zero.
"""
init_state() = NaceState(0, Set(), Dict(), Dict(), "Unused", Set(), [])

"""
    (agent::NaceAgent)(obs)

Run a step

Run the complete pipeline from perceiving from the environment observation to
determining the next action to take.
Returns the chosen action, does not have side-effects except for updating the agent's state.
"""
function (agent::NaceAgent)(obs)
    percept_state = agent.perceptor(obs)
    values = map(obj -> obj == "goal" ? 1 : (obj == "lava" ? -1 : 0), percept_state[:BOARD])
    per_ext_ante =
        isempty(agent.state.per_ext_ante) ? percept_state : agent.state.per_ext_ante
    agent.state = NaceState(
        agent.state.t,
        agent.state.focus,
        percept_state,
        per_ext_ante,
        agent.state.act_ante,
        agent.state.rules,
        values,
    )
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
    objects = map(i -> IDX_TO_OBJECT[i], obs["image"][:, :, 1])
    objects = reshape(
        [Cell(Tuple(idx)[1], Tuple(idx)[2], objects[idx]) for idx ∈ eachindex(objects)],
        size(objects),
    )
    Dict(:DIR => obs["direction"], :BOARD => objects, :TASK => obs["mission"])
end

"""
    nace_policy(state)

Run the policy
"""
function nace_policy(state)
    state.act_ante
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

"""
Process an observation
"""
function observe(state::NaceState) end

"""
    hypothesize(state::NaceState)

TBW
"""
function hypothesize(state::NaceState)
    new_rules = Set()
    for c ∈ state.focus
        # Generate new hypotheses
        new_rules = union(new_rules, new_hypotheses(state, c))
    end
    # Initialize containers for rule evidence
    rule_evidence = Dict()
    new_negrules = Set()

    # Update rule evidences
    for rule ∈ new_rules
        if is_valid_rule(rule, state.rules)
            rule_evidence[rule] = true
        else
            push!(new_negrules, rule)
        end
    end

    # Filter rules
    filtered_rules = filter_rules(new_rules, rule_evidence)

    # Return updated focus, rule evidence, new rules, and negative rules
    return state.focus, rule_evidence, filtered_rules, new_negrules
end

"""
    new_hypotheses(agent_state::NaceState, perceived_externals, previous_externals, action, c3)

# Arguments

  - c3: the cell to be used for the consequence.
"""
function new_hypotheses(agent_state::NaceState, c3)
    percv_ext = agent_state.perceived_externals
    previous_externals = agent_state.per_ext_ante
    action = agent_state.act_ante
    new_rules = Set()

    board_ante = previous_externals[:BOARD]
    board = percv_ext[:BOARD]
    for pos1 ∈ zip(1:size(board), 1:size(board))
        for pos2 ∈ zip(1:size(board), 1:size(board))
            param_name = :BOARD
            c1 = board_ante[param_name][pos1]  # one precondition cell
            c2 = board_ante[param_name][pos2]  # another precondition cell
            potential_rule = generate_rule(agent_state, param_name, c1, c2, c3, action)
            push!(new_rules, potential_rule)
        end
    end
    return new_rules
end

function filter_rules(new_rules, rule_evidence)
    filtered_rules = Set()
    for rule ∈ new_rules
        if rule_evidence[rule] && !conflicting_rule_exists(rule, new_rules)
            push!(filtered_rules, rule)
        end
    end
    return filtered_rules
end

function conflicting_rule_exists(rule, rules)
    for existing_rule ∈ rules
        if existing_rule.precondition == rule.precondition &&
           existing_rule.consequence != rule.consequence
            return true
        end
    end
    return false
end

"""
    generate_rule(param_name, old_value, new_value, action)

Generate a new rule

# Arguments

  - `param_name` :: Symbol: The name of the perceived state parameter (`BOARD`, `VALUES`, `DIR`).
"""
function generate_rule(agent_state::NaceState, param_name::Symbol, pos1, pos2, pos3, action)
    # Define or obtain the values for cell1, cell2, agent_state, and cell
    percv_ext = agent_state.per_ext_ante
    board = percv_ext[:BOARD]
    cell1 = board[pos1]
    cell2 = board[pos2]
    cell3 = board[pos3]
    reward = percv_ext[:REWARD]

    # Create Precondition with appropriate values
    precondition = Precondition(cell1, cell2, agent_state, action)

    # Create Consequence with appropriate values
    consequence = Consequence(cell3, agent_state, reward)

    # Initialize evidence and scores
    evidence_pos = Int32(0)
    evidence_neg = Int32(0)
    score = 0.0f0
    acc_score = 0.0f0

    # Return Rule with all required arguments
    return Rule(precondition, consequence, evidence_pos, evidence_neg, score, acc_score)
end

function is_valid_rule(rule, existing_rules)
    for existing_rule ∈ existing_rules
        if existing_rule.precondition == rule.precondition &&
           existing_rule.consequence == rule.consequence
            return false
        end
    end
    return true
end

"""
    predict(state::NaceState, grid_width::Int, grid_height::Int)

Apply rules to predict the future world state. (TODO: explain more)
"""
function predict(state::NaceState, grid_width::Int, grid_height::Int)
    per_ext_post = deepcopy(state.per_ext_ante)
    used_rules_sumscore = 0.0f0
    used_rules_amount = 0
    position_scores, highest_highscore = filter_hypotheses(grid_width, grid_height, state)
    age = 0
    max_focus = maximum(identity, state.focus; init=0)

    for rule ∈ state.rules
        for (x, y) ∈ keys(position_scores)
            cell = state.perceived_externals[:BOARD][(x, y)]
            scores, highscore, rule = position_scores[(x, y)]

            if applicable(state_value(state), rule_ratio(cell, rule))
                per_ext_post[:BOARD][(x, y)] = rule.consequence.cell.item
            end
        end
    end

    # if !isempty(state.values)
    #     if state.values[begin] == 1 && score == 1.0
    #         score = -Inf32
    #     elseif state.values[begin] == -1 && score == 1.0
    #         score = Inf32
    #     end
    # else
    #     score = Inf32
    # end

    return per_ext_post
end

"""
    filter_hypotheses(width::Int, height::Int, state::NaceState)

Filter hypotheses down to ones with high enough quality
"""
function filter_hypotheses(state::NaceState)
    width, height = size(state.perceived_externals[:BOARD])
    attend_positions = Set{Tuple{Int,Int}}()
    position_scores = Dict{Tuple{Int,Int},Any}()
    highest_highscore = 0.0f0

    for y ∈ 1:height
        for x ∈ 1:width
            if state.per_ext_ante[:BOARD][x, y] in state.focus
                push!(attend_positions, (x, y))
                for rule ∈ state.rules
                    action_score_and_preconditions = collect(rule.precondition)
                    for (x_rel, y_rel, required_state) ∈
                        action_score_and_preconditions[3:end]
                        push!(attend_positions, (x + x_rel, y + y_rel))
                    end
                end
            end
        end
    end

    for (x, y) ∈ attend_positions
        scores = Dict{Any,Float64}()
        position_scores[(x, y)] = scores
        highscore = 0.0f0
        highscore_rule = nothing
        for rule ∈ state.rules
            precondition, consequence = rule
            action_score_and_preconditions = collect(precondition)
            values = action_score_and_preconditions[2]
            if action_score_and_preconditions[1] == state.act_ante
                scores[rule] = 0.0f0
            else
                continue
            end
            continue_flag = false
            for i ∈ eachindex(values)
                if values[i] != state.per_ext_ante[:VALUES][i+1]
                    continue_flag = true
                    break
                end
            end
            if continue_flag
                continue
            end
            for (x_rel, y_rel, required_state) ∈ action_score_and_preconditions[3:end]
                if y + y_rel > height || y + y_rel < 1 || x + x_rel > width || x + x_rel < 1
                    continue_flag = true
                    break
                end
                if state.per_ext_ante[:BOARD][x+x_rel][y+y_rel] == required_state
                    scores[rule] += 1.0
                end
            end
            if continue_flag
                continue
            end
            scores[rule] /= length(precondition) - 2
            if scores[rule] > 0.0 &&
               (scores[rule] > highscore || (scores[rule] == highscore &&
                 !isnothing(highscore_rule) &&
                 length(rule[1]) > length(highscore_rule[1])))
                highscore = get(scores, rule, 0.0)
                highscore_rule = rule
            end
        end
        position_scores[(x, y)] = (scores, highscore, highscore_rule)
        if highscore > highest_highscore
            highest_highscore = highscore
        end
    end

    position_scores, highest_highscore
end

"""
    plan(state::NaceState, actions, max_depth::Int, max_queue_len::Int, custom_goal)

Plan and choose best actions to take. (TODO: clarify / explain)
"""
function plan(state::NaceState, actions, max_depth::Int, max_queue_len::Int, custom_goal)
    # Initialize variables
    best_actions = []
    best_score = Inf32
    best_action_combination_for_revisit = []
    oldest_age = 0.0f0

    # 1. Search for argmax V(s) > 0
    best_actions, best_score =
        bfs_with_predictor(state, actions, max_depth, max_queue_len, :argmax)
    if !isempty(best_actions)
        return best_actions, best_score, best_action_combination_for_revisit, oldest_age
    end

    # 2. Search for argmin S(s) < 1
    best_actions, best_score =
        bfs_with_predictor(state, actions, max_depth, max_queue_len, :argmin)
    if !isempty(best_actions)
        return best_actions, best_score, best_action_combination_for_revisit, oldest_age
    end

    # 3. Random action
    random_action = [rand(actions)]
    return random_action, best_score, best_action_combination_for_revisit, oldest_age
end

function bfs_with_predictor(
    state::NaceState,
    actions,
    max_depth::Int,
    max_queue_len::Int,
    mode::Symbol,
)
    queue = Queue{Tuple{NaceState,Vector{String},Int}}()
    enqueue!(queue, (state, [], 0))
    best_actions = []
    best_score = mode == :argmax ? -Inf32 : Inf32

    while !isempty(queue)
        current_state, action_sequence, depth = dequeue!(queue)

        if depth > max_depth
            continue
        end

        per_ext_post, score, age, _ = predict(current_state, 7, 7)
        # Implement logic to apply rules with Q(r, c) = 1 and maximum truthexp(r)
        # Ensure the predicted state is constructed correctly
        predicted_state = NaceState(
            current_state.t + 1,
            current_state.focus,
            per_ext_post,
            current_state.perceived_externals,
            current_state.act_ante,
            current_state.rules,
        )

        if (mode == :argmax && score > 0) || (mode == :argmin && score < 1)
            if (mode == :argmax && score > best_score) ||
               (mode == :argmin && score < best_score)
                best_actions = action_sequence
                best_score = score
            end
        end

        for action ∈ actions
            new_state = deepcopy(predicted_state)
            new_action_sequence = vcat(action_sequence, [action])
            enqueue!(queue, (new_state, new_action_sequence, depth + 1))
        end

        if length(queue) > max_queue_len
            break
        end
    end

    return best_actions, best_score
end

"""
    cycle(state::NaceState)

Perform a single agent cycle

An agent cycle consists of running all the previously defined logic to produce
information necessary to update its state and select the next action to take for one timestep.
"""
function cycle(state::NaceState)::NaceState
    # Predict the next state
    new_world = predict(state, 7, 7)

    # Hypothesize new rules
    focus, rule_evidence, new_rules, new_negrules = hypothesize(state)

    # Plan the next actions
    planned_actions = plan(state, keys(ACTION_TO_IDX), 100, 2000, nothing)

    # Determine the next action
    action = isempty(planned_actions) ? IDX_TO_ACTION[rand(0:7-1)] : planned_actions[1]

    # Return the updated state
    return NaceState(
        state.t + 1,
        focus,
        new_world,
        state.perceived_externals,
        action,
        union(state.rules, new_rules),
        state.values,
    )
end

end # module
