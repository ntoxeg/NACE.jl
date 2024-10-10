module NACE
export make_random_policy, run_example, run_example_random

using DataStructures
using PyCall

function __init__()
    global gym = pyimport("gymnasium")
    global miniwrap = pyimport("minigrid.wrappers")
end

# TODO: refactor into those files
include("base.jl")
include("envs.jl")
include("agent.jl")

# example of running in the REPL
# env = NACE.gym.make("MiniGrid-LavaCrossingS11N5-v0", render_mode="human");
# obs, info = env.reset();

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
    NaceState(t, focus, perceived_externals, per_ext_ante, act_ante, rules)

Agent state structure

# Arguments

  - `t` :: Int: Current time step.
  - `focus` :: Set: Set of objects the agent is currently focused on.
  - `perceived_externals` :: Dict: Perceived external state, including objects, walls, and agents.
  - `per_ext_ante` :: Dict: Previous perceived external state from the previous time step.
  - `act_ante` :: String: Action taken in the previous time step.
  - `rules` :: Set: Set of rules that the agent is currently believes.
"""
struct NaceState
    t::Int
    focus::Set
    perceived_externals::Dict
    per_ext_ante::Dict
    act_ante::String
    rules::Set
end

"""
    init_state()

Create an empty state with time step zero.
"""
init_state() = NaceState(0, Set(), Dict(), Dict(), "", Set())

"""
    NaceAgent(state, policy, perceptor, effector)

Non-Axiomatic Causal Explorer agent.

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

"""
    (agent::NaceAgent)(obs)

Run a step.

Run the complete pipeline from perceiving from the environment observation to
determining the next action to take.
Returns the chosen action, does not have side-effects except for updating the agent's state.
"""
function (agent::NaceAgent)(obs)
    percept_state = agent.perceptor(obs)
    per_ext_ante =
        isempty(agent.state.per_ext_ante) ? percept_state : agent.state.per_ext_ante
    agent.state = NaceState(
        agent.state.t,
        agent.state.focus,
        percept_state,
        per_ext_ante,
        agent.state.act_ante,
        agent.state.rules,
    )
    agent.state = agent.policy(agent.state)
    agent.effector(agent.state.act_ante)
end

"""
    nace_perceptor(obs)

Run the perceptor.

Run the perceptor -- the function that consumes an environment observation
data structure and returns a representation usable within the agent's internal
logic.
"""
function nace_perceptor(obs)
    objects = map(i -> IDX_TO_OBJECT[i], obs["image"][:, :, 1])
    Dict(
        :DIR => obs["direction"],
        :BOARD => objects,
        :VALUES => map(obj -> obj == "goal" ? 1 : (obj == "lava" ? -1 : 0), objects), # FIXME
        :TASK => obs["mission"],
    )
end

"""
    nace_policy(state)

Run the policy.
"""
function nace_policy(state)
    focus, rules, action, _... = cycle(state)
    NaceState(
        state.t + 1,
        focus,
        state.perceived_externals,
        state.perceived_externals,
        action,
        rules,
    )
end

"""
    nace_effector(action)

Run the effector.

Run the effector -- the function that takes an action as input and returns its representation
usable with the environment's API.
"""
function nace_effector(action)
    ACTION_TO_IDX[action]
end

"""
    run_example(env)

Run an example on an environment.
"""
function run_example(env)
    obs, info = env.reset()
    agent = NaceAgent(
        NaceState(0, Set(), Dict(), Dict(), "Unused", Set()),
        nace_policy,
        nace_perceptor,
        nace_effector,
    )
    for _ ∈ 1:10
        action = agent(obs)
        println("Action: $(IDX_TO_ACTION[action])")
        obs, info = env.step(action)
        println("Info: $info")
    end
end

"""
    hypothesize(state::NaceState)

TBW
"""
function hypothesize(state::NaceState)
    # Extract current focus and perceived externals
    focus = state.focus
    perceived_externals = state.perceived_externals
    previous_externals = state.per_ext_ante

    # Generate new hypotheses
    new_rules = new_hypotheses(perceived_externals, previous_externals, state.act_ante)

    # Initialize containers for rule evidence
    rule_evidence = Dict()
    new_negrules = Set()

    # Update rule evidences
    for rule in new_rules
        if is_valid_rule(rule, state.rules)
            rule_evidence[rule] = true
        else
            push!(new_negrules, rule)
        end
    end

    # Filter rules
    filtered_rules = filter_rules(new_rules, rule_evidence)

    # Return updated focus, rule evidence, new rules, and negative rules
    return focus, rule_evidence, filtered_rules, new_negrules
end

function new_hypotheses(perceived_externals, previous_externals, action)
    new_rules = Set()
    for (key, value) in perceived_externals
        if haskey(previous_externals, key) && previous_externals[key] != value
            potential_rule = generate_rule(key, previous_externals[key], value, action)
            push!(new_rules, potential_rule)
        end
    end
    return new_rules
end

function filter_rules(new_rules, rule_evidence)
    filtered_rules = Set()
    for rule in new_rules
        if rule_evidence[rule] && !conflicting_rule_exists(rule, new_rules)
            push!(filtered_rules, rule)
        end
    end
    return filtered_rules
end

function conflicting_rule_exists(rule, rules)
    # Logic to check for conflicting rules
    return false
end

function generate_rule(key, old_value, new_value, action)
    # Logic to generate a rule based on observed change
    return Rule(Condition("$key == $old_value"), "$key = $new_value", 0.0, 0.0)
end

function is_valid_rule(rule, existing_rules)
    # Logic to validate a rule against existing rules
    return true
end

function prediction_errors() end

function new_hypotheses() end

function verify_hypothesis() end

"""
    predict(state::NaceState, grid_width::Int, grid_height::Int)

Apply rules to predict the future world state. (TODO: explain more)
"""
function predict(state::NaceState, grid_width::Int, grid_height::Int)  # TODO: move gird size args somewhere else
    per_ext_post = deepcopy(state.per_ext_ante)
    used_rules_sumscore = 0.0f0
    used_rules_amount = 0
    position_scores::Dict{Tuple{Int,Int}}, highest_highscore::Float32 =
        filter_hypotheses(grid_width, grid_height, state)
    age = 0
    max_focus = maximum(identity, state.focus; init=0)

    for (x, y) ∈ keys(position_scores)
        scores, highscore, rule = position_scores[(x, y)]

        if applicable(highscore, highest_highscore, rule) # FIXME: scores needed?
            per_ext_post[:BOARD][(x, y)] = rule.consequence # FIXME: rule[2][3] # what is all this
            per_ext_post[:VALUES] = (rule.acc_score, rule.v_inventory) # rule[2][4]
            used_rules_sumscore += rule.score # get(scores, rule, 0.0)
            used_rules_amount += 1
        end

        if max_focus &&
           per_ext_post[:BOARD][(x, y)] in state.focus &&
           per_ext_post[:BOARD][(x, y)] == max_focus
            age = max((state.t - per_ext_post[:TIMES][(x, y)]), age)
        end
    end
    score = used_rules_amount > 0 ? used_rules_sumscore / used_rules_amount : 1.0 # AIRIS confidence
    if !isempty(per_ext_post[:VALUES])
        # but if the certaintly predicted world has higher value, then set prediction score to the best it can be
        if per_ext_post[:VALUES][begin] == 1 && score == 1.0 # || (customGoal && customGoal(per_ext_post))
            score = -Inf32
        end
        # while if the certaintly predicted world has lower value, set prediction score to the worst it can be
        if per_ext_post[:VALUES][begin] == -1 && score == 1.0
            score = Inf32
        end
    else
        score = Inf32
    end
    per_ext_post, score, age, per_ext_post[:VALUES]
end

"""
    filter_hypotheses(width::Int, height::Int, state::NaceState)

Filter hypotheses down to ones with high enough quality. (TODO: clarify)
"""
function filter_hypotheses(width::Int, height::Int, state::NaceState)
    attend_positions = Set{Tuple{Int,Int}}()
    position_scores = Dict{Tuple{Int,Int},Any}()
    highest_highscore = 0.0f0

    for y ∈ 1:height
        for x ∈ 1:width
            if state.per_ext_ante[:BOARD][x, y] in state.focus
                push!(attend_positions, (x, y))
                for rule ∈ state.rules
                    precondition, consequence = rule
                    action_score_and_preconditions = collect(precondition)
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

function max_truth_exp() end

function best_hypothesis() end

"""
    plan(state::NaceState, actions, max_depth::Int, max_queue_len::Int, custom_goal)

Plan and choose best actions to take. (TODO: clarify / explain)
"""
function plan(state::NaceState, actions, max_depth::Int, max_queue_len::Int, custom_goal)
    queue = Queue{Tuple{NaceState,Vector,Int}}() # state, action list, depth
    enqueue!(queue, (state, [], 0))

    encountered_at = Dict()
    best_score = Inf32
    best_actions = []
    best_action_combination_for_revisit = []
    oldest_age = 0.0f0

    while !isempty(queue)
        if length(queue) > max_queue_len
            println("Planning queue bound enforced!")
            break
        end
        current_state, planned_actions, depth = dequeue!(queue)  # Dequeue
        if depth > max_depth  # If maximum depth is reached, stop searching
            continue
        end
        world_BOARD_VALUES = state.perceived_externals[:VALUES][1] # TODO: figure this out FIXME
        # println(world_BOARD_VALUES) TODO: remove debug prints
        # println(encountered_at)
        # println(depth)
        if depth >= get(encountered_at, world_BOARD_VALUES, Inf)
            continue
        else
            encountered_at[world_BOARD_VALUES] = depth
        end
        for action ∈ actions
            new_world, new_score, new_age, _ = predict(state, 7, 7)
            if new_world == current_state || new_score == Inf32
                continue
            end
            new_planned_actions = [planned_actions; action]
            if new_score < best_score ||
               (new_score == best_score && size(new_planned_actions) < size(best_actions))
                best_actions = new_planned_actions
                best_score = new_score
            end
            if new_age > oldest_age || (new_age == oldest_age &&
                size(new_planned_actions) < size(best_action_combination_for_revisit))
                best_action_combination_for_revisit = new_planned_actions
                oldest_age = new_age
            end
            if new_score == 1.0
                enqueue!(queue, (state, new_planned_actions, depth + 1))  # Enqueue children FIXME new_world
            end
            if new_score == -Inf32
                queue = Queue{Tuple{NaceState,Vector,Int}}()
                break
            end
        end
    end

    best_actions, best_score, best_action_combination_for_revisit, oldest_age
end

function highest_reward() end

function weakest_hypothesis() end

function oldest_observed() end

"""
    cycle(state::NaceState)

Perform a single agent cycle.

An agent cycle consists of running all the previously defined logic to produce
information necessary to update its state and select the next action to take for one timestep.
"""
function cycle(state::NaceState)#::Tuple{Set,String} FIXME
    # prediction L 116
    # predict for the rest of the plan L 130
    # moved from below

    custom_goal = nothing
    rules = state.rules
    rules_exclude = Set()
    # next obs needed (L 109)
    # we need observation diff as well
    # diff state.perceived_externals vs state.per_ext_ante
    new_world, new_score, new_age, _ = predict(state, 7, 7)

    focus, rule_evidence, new_rules, new_negrules = hypothesize(state)
    # add "excluded" rules back TODO
    fav_actions, airis_score, fav_actions_revisit, oldest_age =
        plan(state, keys(ACTION_TO_IDX), 100, 2000, custom_goal)
    # mode selection & babbling
    # modes: EXPLORE, BABBLE, ACHIEVE, CURIOUS

    # post-block effects: next action && plan determined.
    # action enaction, break here

    action = isempty(fav_actions) ? IDX_TO_ACTION[rand(0:7-1)] : fav_actions[begin] # TODO: proper impl
    lastplanworld = new_world
    per_ext_post = state.perceived_externals # FIXME
    behavior = nothing

    focus,
    new_rules,
    action,
    rule_evidence,
    new_rules,
    new_negrules,
    lastplanworld,
    per_ext_post,
    values,
    behavior,
    plan
end

end # module
