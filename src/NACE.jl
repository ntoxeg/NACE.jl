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
init_state() = NaceState(0, Set(), Dict(), Dict(), "Unused", Set{Rule}(), Vector{Int}())

"""
    (agent::NaceAgent)(obs)

Run a step

Run the complete pipeline from perceiving from the environment observation to
determining the next action to take.
Returns the chosen action, does not have side-effects except for updating the agent's state.
"""
function (agent::NaceAgent)(obs)
    percept_state = agent.perceptor(obs)
    values = percept_state[:VALUES]
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
        [Cell(idx[1], idx[2], objects[idx]) for idx ∈ CartesianIndices(objects)],
        size(objects),
    )
    values = map(obj -> obj == "goal" ? 1 : (obj == "lava" ? -1 : 0), vec(objects))
    Dict(
        :DIR => obs["direction"],
        :BOARD => objects,
        :TASK => obs["mission"],
        :VALUES => values,
    )
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
    height, width = size(board)

    for i ∈ 1:height, j ∈ 1:width
        for k ∈ 1:height, l ∈ 1:width
            param_name = :BOARD
            c1 = board_ante[i, j]  # one precondition cell
            c2 = board_ante[k, l]  # another precondition cell
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
function generate_rule(
    agent_state::NaceState,
    param_name::Symbol,
    cell1::Cell,
    cell2::Cell,
    cell3,
    action,
)
    # Create Precondition with appropriate values
    precondition = Precondition(cell1, cell2, agent_state, action)

    # Create Consequence with appropriate values
    consequence = Consequence(Cell(cell1.x, cell1.y, cell3), agent_state, 0)  # Initialize reward as 0

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

    # Get the best rules for prediction
    best_rules = choose_rules(state.rules)
    position_scores = Dict{Tuple{Int,Int},Any}()
    highest_highscore = 0.0f0

    # Calculate scores for each position
    for x ∈ 1:grid_width, y ∈ 1:grid_height
        cell = state.perceived_externals[:BOARD][x, y]
        scores = Dict{Rule,Float32}()
        highscore = 0.0f0
        highscore_rule = nothing

        for rule ∈ best_rules
            # Calculate rule applicability score
            score = rule_ratio(cell, rule)
            if score > 0.0f0
                scores[rule] = score
                if score > highscore
                    highscore = score
                    highscore_rule = rule
                end
            end
        end

        position_scores[(x, y)] = (scores, highscore, highscore_rule)
        highest_highscore = max(highest_highscore, highscore)
    end

    # Apply the best rules to predict the next state
    for (pos, (scores, highscore, rule)) ∈ position_scores
        if !isnothing(rule) && applicable(state_value(state), scores[rule])
            x, y = pos
            per_ext_post[:BOARD][x, y] = rule.consequence.cell.item
            used_rules_sumscore += rule.score
            used_rules_amount += 1
        end
    end

    # Update rule scores based on usage
    if used_rules_amount > 0
        avg_score = used_rules_sumscore / used_rules_amount
        for rule ∈ best_rules
            rule.acc_score += avg_score
        end
    end

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
    best_score = -Inf32
    best_action_combination_for_revisit = []
    oldest_age = 0.0f0

    # Get the best rules for planning
    best_rules = choose_rules(state.rules)

    # If we have good rules, use them for planning
    if !isempty(best_rules)
        best_actions, best_score =
            bfs_with_predictor(state, actions, max_depth, max_queue_len, :argmax)
        if !isempty(best_actions)
            return best_actions, best_score, best_action_combination_for_revisit, oldest_age
        end
    end

    # If no good rules found, try exploration
    best_actions, best_score =
        bfs_with_predictor(state, actions, max_depth, max_queue_len, :argmin)
    if !isempty(best_actions)
        return best_actions, best_score, best_action_combination_for_revisit, oldest_age
    end

    # If all else fails, use the oldest rule for guidance
    oldest_rule = oldest_observed(state.rules, max_depth)
    if !isnothing(oldest_rule)
        oldest_age = oldest_rule.acc_score
        action_sequence = [oldest_rule.precondition.action]
        return action_sequence, 0.0f0, action_sequence, oldest_age
    end

    # Last resort: random action
    random_action = [rand(actions)]
    return random_action, 0.0f0, [], oldest_age
end

function bfs_with_predictor(
    state::NaceState,
    actions,
    max_depth::Int,
    max_queue_len::Int,
    mode::Symbol,
)
    queue = Queue{Tuple{NaceState,Vector{String},Int,Float32}}()
    enqueue!(queue, (state, [], 0, 0.0f0))
    best_actions = []
    best_score = mode == :argmax ? -Inf32 : Inf32

    while !isempty(queue)
        current_state, action_sequence, depth, current_score = dequeue!(queue)

        if depth > max_depth
            continue
        end

        # Predict next state
        per_ext_post = predict(current_state, 7, 7)

        # Calculate score based on rules and current state
        score = 0.0f0
        for rule ∈ current_state.rules
            if rule_active(rule)
                score += rule.score * truthexp(rule)
            end
        end

        # Update best score and actions
        if (mode == :argmax && score > best_score) ||
           (mode == :argmin && score < best_score)
            best_actions = action_sequence
            best_score = score
        end

        # Add new states to queue
        for action ∈ actions
            if action != "Unused"
                new_state = NaceState(
                    current_state.t + 1,
                    current_state.focus,
                    per_ext_post,
                    current_state.perceived_externals,
                    action,
                    current_state.rules,
                    current_state.values,
                )
                new_score = current_score + score
                enqueue!(
                    queue,
                    (new_state, vcat(action_sequence, [action]), depth + 1, new_score),
                )
            end
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
    # Update rule evidence based on current observations
    if !isempty(state.per_ext_ante)
        M_change, M_observation_mismatched, M_prediction_mismatched =
            calculate_sets(state, state)
        rule_memory = RuleMemory(state.rules)
        update_rule_evidence(
            rule_memory,
            M_change,
            M_observation_mismatched,
            M_prediction_mismatched,
        )
    end

    # Predict the next state
    new_world = predict(state, 7, 7)

    # Hypothesize new rules
    focus, rule_evidence, new_rules, new_negrules = hypothesize(state)

    # Update focus based on prediction mismatches and changes
    if !isempty(state.per_ext_ante)
        M_change, _, M_prediction_mismatched = calculate_sets(state, state)
        focus = union(
            focus,
            Set(cell.item for cell ∈ M_change),
            Set(cell.item for cell ∈ M_prediction_mismatched),
        )
    end

    # Plan the next actions using the improved planning system
    planned_actions, score, revisit_actions, age =
        plan(state, keys(ACTION_TO_IDX), 100, 2000, nothing)

    # Determine the next action, preferring planned actions over random ones
    action = if !isempty(planned_actions)
        planned_actions[1]
    else
        # If no planned actions, use the action from the best rule
        best_rule = max_truth_exp(state.rules)
        if !isnothing(best_rule)
            best_rule.precondition.action
        else
            # Fallback to random action if no good rules exist
            IDX_TO_ACTION[rand(0:6)]
        end
    end

    # Update rule scores based on the chosen action
    for rule ∈ state.rules
        if rule.precondition.action == action
            rule.score += 0.1f0  # Small positive reinforcement for chosen action
        end
    end

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
