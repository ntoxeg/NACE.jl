"""
Debug levels:
0 = No debug output
1 = Basic state changes and important events
2 = Detailed rule generation and evidence updates
3 = Very detailed cell-by-cell analysis
"""
const DEBUG_LEVEL = Ref(0)

"""
    debug_print(level::Int, msg::String)

Print debug message if current debug level is >= specified level.
"""
function debug_print(level::Int, msg::String)
    if DEBUG_LEVEL[] >= level
        println(msg)
    end
end

export Rule,
    rule_applicable,
    Cell,
    NaceState,
    rule_ratio,
    cell_value,
    state_value,
    truthexp,
    make_rule,
    init_state,
    predict,
    calculate_sets,
    hypothesize,
    new_hypotheses,
    cycle,
    AgentContext,
    Precondition,
    Consequence,
    RuleMemory,
    update_rule_evidence,
    choose_rules,
    max_truth_exp,
    best_hypothesis,
    highest_reward,
    plan,
    bfs_with_predictor

struct Cell
    x::Int
    y::Int
    item::String
end

mutable struct AgentContext
    perceived_externals::Dict
    per_ext_ante::Dict
    act_ante::String
end

struct Precondition
    cell1::Cell
    cell2::Cell
    context::AgentContext
    action::String
    expr::String
end

struct Consequence
    cell::Cell
    context::AgentContext
    reward::Float32
end

mutable struct Rule
    precondition::Precondition
    consequence::Consequence
    evidence_pos::Float32
    evidence_neg::Float32
    score::Float32
    acc_score::Float32
end

"""
    NaceState(t, focus, rules, values, context)

Agent state structure

# Arguments

  - `t` :: Int: Current time step.
  - `focus` :: Set{Cell}: Set of objects the agent is currently focused on.
  - `rules` :: Set{Rule}: Set of rules that the agent currently believes.
  - `values` :: Vector{Int}: Vector of values associated with the agent's state.
  - `context` :: AgentContext: Contains perceived externals, previous state and action.
"""
struct NaceState
    t::Int
    focus::Set{Cell}
    rules::Set{Rule}
    values::Vector{Int}
    context::AgentContext
end

"""
    init_state()

Create an empty state with time step zero.
"""
init_state() =
    NaceState(0, Set{Cell}(), Set{Rule}(), Vector{Int}(), AgentContext(Dict(), Dict(), ""))

Cell(x::Int, y::Int, item) = Cell(x, y, item)

function cond_match(cond1::Precondition, cond2::Precondition)
    cond1.expr == cond2.expr
end

function truthexp_with(cfun::Function, r::Rule)::AbstractFloat
    w = r.evidence_neg + r.evidence_pos
    if w == 0
        return 0.5f0  # Return neutral value when no evidence
    end
    f = r.evidence_pos / w
    c = cfun(w)
    f * c + 0.5f0 * (1 - c)
end

confidence_count(w) = w / (w + 1)

truthexp = Base.Fix1(truthexp_with, confidence_count)

rule_active(r::Rule)::Bool = r.evidence_pos >= r.evidence_neg

struct RuleMemory
    indeterminate_rules::Set{Rule}
    active_rules::Set{Rule}
    inactive_rules::Set{Rule}

    function RuleMemory(rules::Set{Rule})
        new(rules, Set{Rule}(), Set{Rule}())
    end
end

function update_rule_evidence(
    rulem::RuleMemory,
    M_change,
    M_observation_mismatched,
    M_prediction_mismatched,
)
    learning_rate = 0.1f0
    rules = rulem.indeterminate_rules ∪ rulem.active_rules ∪ rulem.inactive_rules
    m = M_change ∪ M_observation_mismatched
    println("Updating evidence for $(length(rules)) rules")
    println("Changes and mismatches: $(length(m)) cells")

    for rule ∈ rules
        c1 = rule.precondition.cell1
        c2 = rule.precondition.cell2
        c3 = rule.consequence.cell

        if Set([c1, c2, c3]) ⊆ m
            rule.evidence_pos += learning_rate
            println("Rule got positive evidence: $(rule.precondition.expr) -> $(rule.consequence.cell.item)")
            println("New evidence: +$(rule.evidence_pos) -$(rule.evidence_neg)")
        end

        if c3 ∈ M_prediction_mismatched
            rule.evidence_neg += learning_rate
            println("Rule got negative evidence: $(rule.precondition.expr) -> $(rule.consequence.cell.item)")
            println("New evidence: +$(rule.evidence_pos) -$(rule.evidence_neg)")
        end
    end
end

function choose_rules(rules::Set{Rule})
    chosen_rules = Set{Rule}()

    # Get the best rules according to different criteria
    best_truth = max_truth_exp(rules)
    best_evidence = best_hypothesis(rules)
    best_reward = highest_reward(rules)

    # Add the best rules if they exist and are active
    for rule ∈ [best_truth, best_evidence, best_reward]
        if !isnothing(rule) && rule_active(rule)
            push!(chosen_rules, rule)
        end
    end

    chosen_rules
end

function update_bird_view(previous_state, perceived_array)
    # Update the bird view map based on the perceived array
    # Implement logic to update the state
end

function calculate_sets(previous_state::NaceState, current_state::NaceState)
    M_change = Set{Cell}()
    M_observation_mismatched = Set{Cell}()
    M_prediction_mismatched = Set{Cell}()

    # Check if we have the required board states
    if !haskey(previous_state.context.perceived_externals, :BOARD) ||
       !haskey(current_state.context.perceived_externals, :BOARD)
        debug_print(1, "Missing board in externals")
        return M_change, M_observation_mismatched, M_prediction_mismatched
    end

    prev_board = previous_state.context.perceived_externals[:BOARD]
    curr_board = current_state.context.perceived_externals[:BOARD]

    debug_print(3, "Board sizes - prev: $(size(prev_board)), curr: $(size(curr_board))")
    if DEBUG_LEVEL[] >= 3
        debug_print(3, "Previous board contents:")
        for i ∈ 1:size(prev_board, 1)
            row = [prev_board[i, j].item for j ∈ 1:size(prev_board, 2)]
            debug_print(3, join(row, " "))
        end
        debug_print(3, "Current board contents:")
        for i ∈ 1:size(curr_board, 1)
            row = [curr_board[i, j].item for j ∈ 1:size(curr_board, 2)]
            debug_print(3, join(row, " "))
        end
    end

    # Calculate changes between states
    for i ∈ 1:size(prev_board, 1), j ∈ 1:size(prev_board, 2)
        prev_cell = prev_board[i, j]
        curr_cell = curr_board[i, j]

        # Consider a change if:
        # 1. The item type changed
        # 2. A cell became visible (changed from "unseen")
        # 3. A cell became unseen (was visible before)
        if prev_cell.item != curr_cell.item ||
           (prev_cell.item == "unseen" && curr_cell.item != "unseen") ||
           (prev_cell.item != "unseen" && curr_cell.item == "unseen")
            debug_print(
                2,
                "Change detected at ($i,$j): $(prev_cell.item) -> $(curr_cell.item)",
            )
            push!(M_change, curr_cell)
        end
    end

    # Calculate prediction mismatches
    predicted_state = predict(previous_state, size(prev_board, 1), size(prev_board, 2))
    if haskey(predicted_state, :BOARD)
        predicted_board = predicted_state[:BOARD]
        if DEBUG_LEVEL[] >= 3
            debug_print(3, "Predicted board contents:")
            for i ∈ 1:size(predicted_board, 1)
                row = [predicted_board[i, j].item for j ∈ 1:size(predicted_board, 2)]
                debug_print(3, join(row, " "))
            end
        end

        for i ∈ 1:size(curr_board, 1), j ∈ 1:size(curr_board, 2)
            pred_cell = predicted_board[i, j]
            curr_cell = curr_board[i, j]

            # Consider a prediction mismatch if:
            # 1. The predicted item is different from the current item
            # 2. We predicted unseen but got a visible cell
            # 3. We predicted a visible cell but got unseen
            if pred_cell.item != curr_cell.item ||
               (pred_cell.item == "unseen" && curr_cell.item != "unseen") ||
               (pred_cell.item != "unseen" && curr_cell.item == "unseen")
                debug_print(
                    2,
                    "Prediction mismatch at ($i,$j): predicted $(pred_cell.item), got $(curr_cell.item)",
                )
                push!(M_prediction_mismatched, curr_cell)
            end
        end
    else
        debug_print(1, "No board in predicted state")
    end

    # Calculate observation mismatches
    for cell ∈ M_change
        if cell in M_prediction_mismatched
            debug_print(2, "Observation mismatch at ($(cell.x),$(cell.y))")
            push!(M_observation_mismatched, cell)
        end
    end

    debug_print(
        1,
        "Set sizes - changes: $(length(M_change)), pred mismatches: $(length(M_prediction_mismatched)), obs mismatches: $(length(M_observation_mismatched))",
    )
    return M_change, M_observation_mismatched, M_prediction_mismatched
end

function hypothesize(state::NaceState)
    # Filter focus to ensure only Cells are present
    filtered_focus = Set{Cell}(filter(x -> x isa Cell, state.focus))
    println("Focus size: $(length(filtered_focus))")

    new_rules = Set{Rule}()
    for c ∈ filtered_focus
        # Generate new hypotheses using properly-typed Cell
        rules = new_hypotheses(state, c)
        println("Generated $(length(rules)) rules for cell at ($(c.x),$(c.y))")
        new_rules = union(new_rules, rules)
    end

    # Initialize containers for rule evidence
    rule_evidence = Dict{Rule,Bool}()
    new_negrules = Set{Rule}()

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
    println("Final rules - new: $(length(filtered_rules)), negative: $(length(new_negrules))")

    # Return updated focus (filtered), rule evidence, filtered new rules, and negative rules
    return filtered_focus, rule_evidence, filtered_rules, new_negrules
end

function Base.show(io::IO, rule::Rule)
    print(
        io,
        "Rule[\n",
        "Precondition: ",
        rule.precondition.expr,
        ",\n",
        "Consequence: ",
        rule.consequence.cell.item,
        ",\n",
        "Score: ",
        rule.score,
        "\n]",
    )
end

function format_rule_comp(key::AbstractString, comp::AbstractString)
    rows = split(comp, ";")
    array_rows = map(row -> split(strip(row)), rows)
    prefix = "$key = "
    if key == "VALUES"
        convert_row_int(row) = map(el -> parse(Int32, String(el)), row)
        array_rows = map(row -> convert_row_int(row), array_rows)
        prefix = "$key = \n"
    elseif key == "BOARD"
        convert_row_str(row) = map(el -> replace(el, "\"" => ""), row)
        array_rows = map(row -> convert_row_str(row), array_rows)
        prefix = "$key = \n"
    elseif key == "DIR"
        convert_dir(row) = map(el -> parse(Int32, String(el)), row)
        array_rows = map(row -> convert_dir(row), array_rows)
    else
        println("[WARNING] Unknown key: $key")
    end
    data = length(array_rows) > 1 ? stack(array_rows) : array_rows[1][1]
    prefix * repr("text/plain", data)
end

function format_2d_array(s::AbstractString)
    comps = split(s, "=")
    if length(comps) < 2
        return s
    end
    fmtstr = format_rule_comp(strip(comps[1]), comps[2])
    if length(comps) == 4
        fmtstr2 = format_rule_comp(strip(comps[3]), comps[4])
        return fmtstr, fmtstr2
    else
        return fmtstr
    end
end

Base.show(io::IO, cond::Precondition) = print(io, cond.expr)

function Base.show(io::IO, c::Consequence)
    print(io, c.cell.item)
end

function rule_empty()
    empty_context = AgentContext(Dict(), Dict(), "")
    empty_state = NaceState(0, Set{Cell}(), Set{Rule}(), Vector{Int}(), empty_context)
    precond = Precondition(Cell(0, 0, ""), Cell(0, 0, ""), empty_context, "", "empty")
    conseq = Consequence(Cell(0, 0, ""), empty_context, 0.0f0)
    Rule(precond, conseq, 0.0f0, 0.0f0, 0.0f0, 0.0f0)
end

"""
    rule_ratio(c::Cell, r::Rule)

Calculate the match ratio of a rule for a given cell.
"""
function rule_ratio(c::Cell, r::Rule)
    c.item == r.consequence.cell.item ? 1.0f0 : 0.0f0
end

"""
    cell_value(rs::Set{Rule}, c::Cell)

Calculate the match value of a cell.

The match value of a cell is the maximum of rule match ratios, for all possible rules.
If there are no rules, returns 0.0f0.
"""
function cell_value(rs::Set{Rule}, c::Cell)
    isempty(rs) && return 0.0f0
    maximum(map(r -> rule_ratio(c, r), collect(rs)))
end

"""
    state_value(s::NaceState)

Calculate the match value of a state.

The state match value is the maximum match value of all cells in the state.
If there are no cells or rules, returns 0.0f0.
"""
function state_value(s::NaceState)
    board = s.context.perceived_externals[:BOARD]
    isempty(board) && return 0.0f0
    isempty(s.rules) && return 0.0f0
    maximum(map(c -> cell_value(s.rules, c), collect(board)))
end

"""
    rule_applicable(sv::Float64, rr::Float64)::Bool

Determine whether a rule is applicable based on its match ratio relative to the state value.
"""
function rule_applicable(sv::Float32, rr::Float32)::Bool
    rr > 0.0 && rr == sv
end

"""
    max_truth_exp(rules::Set{Rule})

Find the rule with the maximum truth expectation.
"""
function max_truth_exp(rules::Set{Rule})
    isempty(rules) && return nothing
    max_rule = nothing
    max_exp = -Inf32
    for rule ∈ rules
        exp = truthexp(rule)
        if exp > max_exp
            max_exp = exp
            max_rule = rule
        end
    end
    max_rule
end

"""
    best_hypothesis(rules::Set{Rule})

Find the hypothesis with the highest positive evidence.
"""
function best_hypothesis(rules::Set{Rule})
    isempty(rules) && return nothing
    max_rule = nothing
    max_evidence = -Inf32
    for rule ∈ rules
        if rule.evidence_pos > max_evidence
            max_evidence = rule.evidence_pos
            max_rule = rule
        end
    end
    max_rule
end

"""
    highest_reward(rules::Set{Rule})

Find the rule that leads to the highest reward.
"""
function highest_reward(rules::Set{Rule})
    isempty(rules) && return nothing
    max_rule = nothing
    max_reward = -Inf32
    for rule ∈ rules
        if rule.consequence.reward > max_reward
            max_reward = rule.consequence.reward
            max_rule = rule
        end
    end
    max_rule
end

"""
    weakest_hypothesis(rules::Set{Rule})

Find the hypothesis with the lowest evidence support.
"""
function weakest_hypothesis(rules::Set{Rule})
    isempty(rules) && return nothing
    min_rule = nothing
    min_evidence = Inf32
    for rule ∈ rules
        total_evidence = rule.evidence_pos + rule.evidence_neg
        if total_evidence < min_evidence
            min_evidence = total_evidence
            min_rule = rule
        end
    end
    min_rule
end

"""
    oldest_observed(rules::Set{Rule}, max_age::Int)

Find the oldest rule that hasn't been observed recently.
"""
function oldest_observed(rules::Set{Rule}, max_age::Int)
    isempty(rules) && return nothing
    oldest_rule = nothing
    max_age_found = -1
    for rule ∈ rules
        # Use accumulated score as a proxy for age
        if rule.acc_score > max_age_found
            max_age_found = rule.acc_score
            oldest_rule = rule
        end
    end
    oldest_rule
end

function make_rule(
    agent_state::NaceState,
    param_name::Symbol,
    cell1::Cell,
    cell2::Cell,
    cell3::Cell,
    action::String,
)
    local expr =
        "if " *
        string(cell1.item) *
        " and " *
        string(cell2.item) *
        " then " *
        string(cell3.item)
    precondition = Precondition(cell1, cell2, agent_state.context, action, expr)
    consequence =
        Consequence(Cell(cell1.x, cell1.y, cell3.item), agent_state.context, 0.0f0)
    evidence_pos = 0.0f0
    evidence_neg = 0.0f0
    score = 0.0f0
    acc_score = 0.0f0
    return Rule(precondition, consequence, evidence_pos, evidence_neg, score, acc_score)
end

"""
    new_hypotheses(agent_state::NaceState, c3::Cell)

Generate new hypotheses for a given cell by looking at its neighborhood.

# Arguments

  - agent_state: Current agent state
  - c3: the cell to be used for the consequence
"""
function new_hypotheses(agent_state::NaceState, c3::Cell)
    percv_ext = agent_state.context.perceived_externals
    previous_externals = agent_state.context.per_ext_ante
    action = agent_state.context.act_ante
    new_rules = Set{Rule}()

    # Check if we have the required board states
    if !haskey(previous_externals, :BOARD) || !haskey(percv_ext, :BOARD)
        debug_print(1, "Missing board in externals")
        return new_rules
    end

    board_ante = previous_externals[:BOARD]
    board = percv_ext[:BOARD]
    height, width = size(board)

    debug_print(
        2,
        "Generating hypotheses for cell at ($(c3.x),$(c3.y)) with item $(c3.item)",
    )
    debug_print(2, "Current action: $action")
    debug_print(2, "Board size: $(height)x$(width)")

    # Look at cells in a radius around the target cell
    radius = 1
    for i ∈ max(1, c3.x - radius):min(height, c3.x + radius)
        for j ∈ max(1, c3.y - radius):min(width, c3.y + radius)
            # Skip the cell itself
            (i == c3.x && j == c3.y) && continue

            # First precondition cell
            c1 = board_ante[i, j]

            # Look for a second cell in the radius
            for k ∈ max(1, c3.x - radius):min(height, c3.x + radius)
                for l ∈ max(1, c3.y - radius):min(width, c3.y + radius)
                    # Skip the first cell and the target cell
                    (k == i && l == j) && continue
                    (k == c3.x && l == c3.y) && continue

                    c2 = board_ante[k, l]
                    debug_print(
                        3,
                        "Considering cells: ($(i),$(j)): $(c1.item), ($(k),$(l)): $(c2.item)",
                    )

                    # Create a rule linking these cells
                    rule = make_rule(agent_state, :BOARD, c1, c2, c3, action)
                    push!(new_rules, rule)
                    debug_print(
                        3,
                        "Generated rule: $(rule.precondition.expr) -> $(rule.consequence.cell.item)",
                    )
                end
            end
        end
    end

    debug_print(2, "Generated $(length(new_rules)) rules total")
    return new_rules
end

function cycle(state::NaceState)::NaceState
    # Initialize empty sets for when we can't update rules
    focus = Set{Cell}()
    new_rules = Set{Rule}()
    M_change = Set{Cell}()
    M_prediction_mismatched = Set{Cell}()

    debug_print(1, "\nStarting cycle at t=$(state.t)")
    debug_print(1, "Initial focus size: $(length(state.focus))")
    debug_print(1, "Initial rules size: $(length(state.rules))")

    # Create a previous state for comparison
    previous_state = NaceState(
        state.t - 1,
        state.focus,
        state.rules,
        state.values,
        AgentContext(state.context.per_ext_ante, Dict(), ""),
    )

    # Update rule evidence based on current observations
    if !isempty(state.context.per_ext_ante)
        debug_print(2, "Previous state exists, calculating changes...")
        M_change, M_observation_mismatched, M_prediction_mismatched =
            calculate_sets(previous_state, state)

        # Add cells around the agent's current position to focus
        if haskey(state.context.perceived_externals, :BOARD)
            board = state.context.perceived_externals[:BOARD]
            height, width = size(board)
            # Find agent position (usually in the center of view)
            agent_x, agent_y = div(height, 2), div(width, 2)
            radius = 1

            # Add cells around agent to focus
            for i ∈ max(1, agent_x - radius):min(height, agent_x + radius)
                for j ∈ max(1, agent_y - radius):min(width, agent_y + radius)
                    push!(M_change, board[i, j])
                end
            end
            debug_print(2, "Added $(length(M_change)) cells around agent to focus")
        end

        rule_memory = RuleMemory(state.rules)
        update_rule_evidence(
            rule_memory,
            M_change,
            M_observation_mismatched,
            M_prediction_mismatched,
        )
    else
        debug_print(1, "No previous state")
    end

    # Predict the next state
    new_world = predict(state, 7, 7)

    # Hypothesize new rules
    try
        debug_print(2, "Attempting to hypothesize new rules...")
        focus, rule_evidence, new_rules, new_negrules = hypothesize(state)
        debug_print(1, "Hypothesized $(length(new_rules)) new rules")
        debug_print(2, "Got $(length(new_negrules)) negative rules")
    catch e
        debug_print(1, "Failed to hypothesize: $e")
        # If hypothesizing fails, keep existing focus and no new rules
        focus = state.focus
        new_rules = Set{Rule}()
    end

    # Update focus based on prediction mismatches and changes
    if !isempty(state.context.per_ext_ante)
        focus = union(
            Set{Cell}(filter(x -> x isa Cell, state.focus)),
            M_change,
            M_prediction_mismatched,
        )
        debug_print(2, "Updated focus size: $(length(focus))")
    end

    # Plan the next actions using the improved planning system
    planned_actions, score, revisit_actions, age =
        plan(state, keys(ACTION_TO_IDX), 100, 2000, nothing)

    # Determine the next action, preferring planned actions over random ones
    action = if !isempty(planned_actions)
        debug_print(1, "Using planned action: $(planned_actions[1])")
        planned_actions[1]
    else
        # If no planned actions, use the action from the best rule
        best_rule = max_truth_exp(state.rules)
        if !isnothing(best_rule)
            debug_print(1, "Using best rule action: $(best_rule.precondition.action)")
            best_rule.precondition.action
        else
            # Fallback to random action if no good rules exist
            # Only use meaningful actions (no "Unused")
            valid_actions =
                filter(a -> !startswith(a, "Unused"), collect(values(IDX_TO_ACTION)))
            random_action = rand(valid_actions)
            debug_print(1, "Using random action: $random_action")
            random_action
        end
    end

    # Update rule scores based on the chosen action
    for rule ∈ state.rules
        if rule.precondition.action == action
            rule.score += 0.1f0  # Small positive reinforcement for chosen action
        end
    end

    # Create new context with updated world state
    new_context = AgentContext(new_world, state.context.perceived_externals, action)

    # Return the updated state
    return NaceState(
        state.t + 1,
        focus,
        union(state.rules, new_rules),
        state.values,
        new_context,
    )
end

"""
    predict(state::NaceState, grid_width::Int, grid_height::Int)

Apply rules to predict the future world state. Returns a copy of the previous external state
if there are no rules or if the state is empty.
"""
function predict(state::NaceState, grid_width::Int, grid_height::Int)
    # If we have no previous state or rules, just return a copy of the current state
    if isempty(state.context.per_ext_ante) || isempty(state.rules)
        return deepcopy(state.context.perceived_externals)
    end

    # Check if BOARD exists in the dictionaries
    if !haskey(state.context.per_ext_ante, :BOARD) ||
       !haskey(state.context.perceived_externals, :BOARD)
        return deepcopy(state.context.perceived_externals)
    end

    per_ext_post = deepcopy(state.context.per_ext_ante)
    used_rules_sumscore = 0.0f0
    used_rules_amount = 0

    # Get the best rules for prediction
    best_rules = choose_rules(state.rules)
    isempty(best_rules) && return per_ext_post

    position_scores = Dict{Tuple{Int,Int},Any}()
    highest_highscore = 0.0f0

    # Calculate scores for each position
    for x ∈ 1:grid_width, y ∈ 1:grid_height
        # Safely get the cell from the board
        cell = try
            state.context.perceived_externals[:BOARD][x, y]
        catch
            continue
        end

        scores = Dict{Rule,Float32}()
        highscore = 0.0f0
        highscore_rule = rule_empty()

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
    for (pos, (scores, highscore, rule::Rule)) ∈ position_scores
        if !isnothing(rule) && rule_applicable(state_value(state), get(scores, rule, 0.0f0))
            x, y = pos
            try
                per_ext_post[:BOARD][x, y] = rule.consequence.cell
                used_rules_sumscore += rule.score
                used_rules_amount += 1
            catch
                continue
            end
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
    filter_rules(rules::Set{Rule}, rule_evidence::Dict{Rule,Bool})

Filter rules based on their evidence.
"""
function filter_rules(rules::Set{Rule}, rule_evidence::Dict{Rule,Bool})
    filtered_rules = Set{Rule}()
    for rule ∈ rules
        if get(rule_evidence, rule, false)
            push!(filtered_rules, rule)
        end
    end
    filtered_rules
end

"""
    is_valid_rule(rule::Rule, existing_rules::Set{Rule})

Check if a rule is valid and not conflicting with existing rules.
"""
function is_valid_rule(rule::Rule, existing_rules::Set{Rule})
    # For now, consider all rules valid if they don't conflict
    !conflicting_rule_exists(rule, existing_rules)
end

"""
    conflicting_rule_exists(rule::Rule, rules::Set{Rule})

Check if there exists a conflicting rule in the set.
"""
function conflicting_rule_exists(rule::Rule, rules::Set{Rule})
    for existing_rule ∈ rules
        if cond_match(rule.precondition, existing_rule.precondition) &&
           rule.consequence.cell.item != existing_rule.consequence.cell.item
            return true
        end
    end
    false
end

"""
    plan(state::NaceState, actions, max_depth::Int, max_queue_len::Int, custom_goal)

Plan a sequence of actions using breadth-first search with prediction.
"""
function plan(state::NaceState, actions, max_depth::Int, max_queue_len::Int, custom_goal)
    # If no rules exist, return empty plan
    if isempty(state.rules)
        return String[], 0.0f0, String[], 0
    end

    # Use BFS with predictor to find a sequence of actions
    actions_seq, score =
        bfs_with_predictor(state, actions, max_depth, max_queue_len, :forward)
    revisit_actions = String[]  # For now, no revisiting
    age = 0  # For now, no age tracking

    return actions_seq, score, revisit_actions, age
end

"""
    bfs_with_predictor(state::NaceState, actions, max_depth::Int, max_queue_len::Int, mode::Symbol)

Perform breadth-first search using the predictor to evaluate states.
"""
function bfs_with_predictor(
    state::NaceState,
    actions,
    max_depth::Int,
    max_queue_len::Int,
    mode::Symbol,
)
    # Initialize queue with current state and empty action sequence
    queue = [(state, String[], 0.0f0)]
    best_score = -Inf32
    best_actions = String[]

    while !isempty(queue) && length(queue) < max_queue_len
        current_state, action_seq, current_score = popfirst!(queue)

        # If we've reached max depth, skip this branch
        length(action_seq) >= max_depth && continue

        # Try each possible action
        for action ∈ actions
            # Create a copy of the state and apply the action
            next_state = deepcopy(current_state)
            next_state.context.act_ante = string(action)

            # Predict the next state
            predicted_world = predict(next_state, 7, 7)  # Using fixed size for now
            next_state.context.perceived_externals = predicted_world

            # Calculate score for this state
            score = state_value(next_state)
            total_score = current_score + score

            # Update best score if this is better
            if total_score > best_score
                best_score = total_score
                best_actions = vcat(action_seq, [string(action)])
            end

            # Add to queue if not at max depth
            if length(action_seq) < max_depth - 1
                push!(queue, (next_state, vcat(action_seq, [string(action)]), total_score))
            end
        end
    end

    return best_actions, best_score
end
