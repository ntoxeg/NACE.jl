using Logging
global_logger(ConsoleLogger(Logging.Info))

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
    Memory,
    Precondition,
    Consequence,
    RuleMemory,
    update_rule_evidence,
    choose_rules,
    max_truth_exp,
    best_hypothesis,
    highest_reward,
    plan,
    bfs_with_predictor,
    write_rules_to_file,
    update_bird_view

struct Cell
    x::Int
    y::Int
    item::String
end

@enum Direction NORTH SOUTH EAST WEST

struct EpisodicMemory
    board::Matrix{Cell}
    direction::Direction
    data::Dict{Symbol,Any}  # Add dictionary for storing additional data
end

# Update constructor to initialize empty data
EpisodicMemory(board::Matrix{Cell}, direction::Direction) =
    EpisodicMemory(board, direction, Dict{Symbol,Any}())

mutable struct Memory
    episodic_current::EpisodicMemory
    episodic_antecedant::EpisodicMemory
    act_ante::String
end

# Add getindex method for EpisodicMemory
Base.getindex(em::EpisodicMemory, key::Symbol) = get(em.data, key, nothing)
Base.setindex!(em::EpisodicMemory, value, key::Symbol) = em.data[key] = value
Base.haskey(em::EpisodicMemory, key::Symbol) = haskey(em.data, key)

# Add iteration methods for EpisodicMemory
Base.iterate(em::EpisodicMemory) = iterate(em.data)
Base.iterate(em::EpisodicMemory, state) = iterate(em.data, state)
Base.isempty(em::EpisodicMemory) = isempty(em.data) && isempty(em.board)
Base.length(em::EpisodicMemory) = length(em.data) + length(em.board)

# Spatial condition with relative coordinates
struct RelativeCondition
    x_offset::Int
    y_offset::Int
    value::String
end

# Internal state values (based of off the paper)
struct ValueTuple
    values::Vector{Float32}
end

struct Precondition
    action::String
    value_tuple::ValueTuple
    conditions::Vector{RelativeCondition}
    expr::String  # Keep for debugging/display
end

struct Consequence
    reward::Float32
    value_tuple::ValueTuple
    cell_value::String
    value_tuple_change::ValueTuple
    target_cell::Cell  # Keep reference to the actual cell
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
  - `memory` :: Memory: Contains perceived externals, previous state and action.
"""
struct NaceState
    t::Int
    focus::Set{Cell}
    rules::Set{Rule}
    values::Vector{Int}
    memory::Memory
end

"""
    init_state()

Create an empty state with time step zero.
"""
init_state() = NaceState(
    0,
    Set{Cell}(),
    Set{Rule}(),
    Vector{Int}(),
    Memory(
        EpisodicMemory(Matrix{Cell}(undef, 0, 0), Direction(0)),
        EpisodicMemory(Matrix{Cell}(undef, 0, 0), Direction(0)),
        "",
    ),
)

function cond_match(cond1::Precondition, cond2::Precondition)
    # Check if actions match
    if cond1.action != cond2.action
        return false
    end

    # Check if conditions match - count matches
    if length(cond1.conditions) != length(cond2.conditions)
        return false
    end

    # Check each condition - need to match all
    for condition1 ∈ cond1.conditions
        condition_found = false
        for condition2 ∈ cond2.conditions
            if condition1.x_offset == condition2.x_offset &&
               condition1.y_offset == condition2.y_offset &&
               condition1.value == condition2.value
                condition_found = true
                break
            end
        end

        if !condition_found
            return false
        end
    end

    # All conditions matched
    return true
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

function update_rule_memory(rulem::RuleMemory)
    # Clear existing categorizations
    empty!(rulem.active_rules)
    empty!(rulem.inactive_rules)

    # Categorize rules based on evidence
    for rule ∈ rulem.indeterminate_rules
        if rule.evidence_pos > rule.evidence_neg
            push!(rulem.active_rules, rule)
        elseif rule.evidence_neg > rule.evidence_pos
            push!(rulem.inactive_rules, rule)
        end
    end

    # Remove categorized rules from indeterminate set
    setdiff!(rulem.indeterminate_rules, union(rulem.active_rules, rulem.inactive_rules))

    return rulem
end

function update_rule_evidence(
    rulem::RuleMemory,
    M_change,
    M_observation_mismatched,
    M_prediction_mismatched,
)
    evidence_mod = 1.0f0
    rules = rulem.indeterminate_rules ∪ rulem.active_rules ∪ rulem.inactive_rules
    m = M_change ∪ M_observation_mismatched
    @info "Updating rule evidence" total_rules = length(rules) changed_cells = length(m)

    for rule ∈ rules
        for condition ∈ rule.precondition.conditions
            # Check each condition relative to the target cell
            # For relative coordinates, we need the actual target cell
            target_cell = rule.consequence.target_cell

            # Calculate absolute coordinates for a precondition cell
            abs_x = target_cell.x + condition.x_offset
            abs_y = target_cell.y + condition.y_offset
            condition_cell = Cell(abs_x, abs_y, condition.value)

            # Check if this cell is in the changed set
            if condition_cell in m
                rule.evidence_pos += evidence_mod
                @debug "Rule got positive evidence" rule = rule.precondition.expr consequence =
                    rule.consequence.cell_value evidence_pos = rule.evidence_pos evidence_neg =
                    rule.evidence_neg
            end
        end

        # Check if the target cell had a prediction mismatch
        if rule.consequence.target_cell ∈ M_prediction_mismatched
            rule.evidence_neg += evidence_mod
            @debug "Rule got negative evidence" rule = rule.precondition.expr consequence =
                rule.consequence.cell_value evidence_pos = rule.evidence_pos evidence_neg =
                rule.evidence_neg
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

function calculate_sets(previous_state::NaceState, current_state::NaceState)
    M_change = Set{Cell}()
    M_observation_mismatched = Set{Cell}()
    M_prediction_mismatched = Set{Cell}()

    # Check if we have the required board states
    if isempty(previous_state.memory.episodic_current.board) ||
       isempty(current_state.memory.episodic_current.board)
        @warn "Missing board in externals"
        return M_change, M_observation_mismatched, M_prediction_mismatched
    end

    prev_board = previous_state.memory.episodic_current.board
    curr_board = current_state.memory.episodic_current.board

    # Calculate changes between states
    for I ∈ CartesianIndices(prev_board)
        prev_cell = prev_board[I]
        curr_cell = curr_board[I]

        # Consider a change significant if:
        # 1. A cell became visible (changed from "unseen" to something else)
        # 2. A visible cell changed its type (e.g., empty to wall)
        # 3. A visible cell became unseen (might indicate movement)
        if (prev_cell.item == "unseen" && curr_cell.item != "unseen") ||
           (prev_cell.item != "unseen" &&
            curr_cell.item != "unseen" &&
            prev_cell.item != curr_cell.item) ||
           (prev_cell.item != "unseen" && curr_cell.item == "unseen")
            @debug "Significant change" position = (I[2], I[1]) from = prev_cell.item to =
                curr_cell.item
            push!(M_change, curr_cell)
        end
    end

    # Calculate prediction mismatches
    predicted_state = predict(previous_state, size(prev_board, 1), size(prev_board, 2))
    if haskey(predicted_state, :BOARD)
        predicted_board = predicted_state[:BOARD]

        for I ∈ CartesianIndices(curr_board)
            pred_cell = predicted_board[I]
            curr_cell = curr_board[I]

            # Consider a prediction mismatch significant if:
            # 1. We predicted a specific item but got something else (both visible)
            # 2. We predicted a visible cell but got unseen
            # 3. We predicted unseen but got a visible cell
            if (pred_cell.item != "unseen" &&
                curr_cell.item != "unseen" &&
                pred_cell.item != curr_cell.item) ||
               (pred_cell.item != "unseen" && curr_cell.item == "unseen") ||
               (pred_cell.item == "unseen" && curr_cell.item != "unseen")
                @debug "Prediction mismatch" position = (I[2], I[1]) predicted =
                    pred_cell.item actual = curr_cell.item
                push!(M_prediction_mismatched, pred_cell)
                @debug "Observation mismatch" position = (curr_cell.x, curr_cell.y)
                push!(M_observation_mismatched, curr_cell)
            end
        end
    end

    @info "Set sizes" changes = length(M_change) pred_mismatches =
        length(M_prediction_mismatched) obs_mismatches = length(M_observation_mismatched)
    return M_change, M_observation_mismatched, M_prediction_mismatched
end

function hypothesize(state::NaceState)
    # Filter focus to ensure only Cells are present
    filtered_focus = Set{Cell}(filter(x -> x isa Cell, state.focus))
    @info "Starting hypothesis generation" focus_size = length(filtered_focus)

    new_rules = Set{Rule}()
    for c ∈ filtered_focus
        # Generate new hypotheses using properly-typed Cell
        rules = new_hypotheses(state, c)
        @debug "Generated rules for cell" position = (c.x, c.y) rule_count = length(rules)
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
    @info "Hypothesis generation complete" new_rules = length(filtered_rules) negative_rules =
        length(new_negrules)

    # Return updated focus (filtered), rule evidence, filtered new rules, and negative rules
    return filtered_focus, rule_evidence, filtered_rules, new_negrules
end

function Base.show(io::IO, rule::Rule)
    # Format conditions for display
    conditions_str = join(
        [
            "$(c.value) at ($(c.x_offset),$(c.y_offset))" for
            c ∈ rule.precondition.conditions
        ],
        " and ",
    )

    print(
        io,
        "Rule[\n",
        "Precondition: ",
        rule.precondition,
        ",\n",
        "Consequence: ",
        rule.consequence,
        "\nEvidence: +$(rule.evidence_pos) -$(rule.evidence_neg)",
        "\nScore: ",
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

function Base.show(io::IO, cond::Precondition)
    # Format conditions for display
    conditions_str = join(
        ["$(c.value) at ($(c.x_offset),$(c.y_offset))" for c ∈ cond.conditions],
        " and ",
    )
    print(io, "if ", conditions_str, " when action is ", cond.action)
end

function Base.show(io::IO, c::Consequence)
    print(io, c.cell_value, " with reward change ", c.reward)
end

"""
    rule_empty()

Create an empty rule for initialization purposes.
"""
function rule_empty()
    # Create an empty relative condition
    empty_condition = RelativeCondition(0, 0, "empty")

    # Create empty value tuples
    empty_values = ValueTuple(Float32[0.0f0])

    # Create empty precondition
    precond = Precondition("", empty_values, [empty_condition], "empty")

    # Create empty consequence
    conseq = Consequence(0.0f0, empty_values, "empty", empty_values, Cell(0, 0, "empty"))

    # Create and return the empty rule
    Rule(precond, conseq, 0.0f0, 0.0f0, 0.0f0, 0.0f0)
end

"""
    rule_ratio(c::Cell, r::Rule)

Calculate the match ratio of a rule for a given cell.
"""
function rule_ratio(c::Cell, r::Rule)::Float32
    nconds = length(r.precondition.conditions)
    n_matched_conds = 0
    for cond ∈ r.precondition.conditions
        if cond.value == c.item
            n_matched_conds += 1
        end
    end
    n_matched_conds / nconds
end

"""
    cell_value(rs::Set{Rule}, c::Cell)

Calculate the match value of a cell.

The match value of a cell is the maximum of rule match ratios, for all possible rules.
If there are no rules, returns 0.0f0.
"""
function cell_value(rs::Set{Rule}, c::Cell)::Float32
    isempty(rs) && return 0.0f0
    maximum(map(r -> rule_ratio(c, r), collect(rs)))
end

"""
    state_value(s::NaceState)

Calculate the match value of a state.

The state match value is the maximum match value of all cells in the state.
If there are no cells or rules, returns 0.0f0.
"""
function state_value(s::NaceState)::Float32
    board = s.memory.episodic_current.board
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
function highest_reward(rules::Set{Rule})::Union{Rule,Nothing}
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
    c3::Cell,  # Consequence cell
    c1::Cell,  # Precondition cell 1
    c2::Cell,  # Precondition cell 2
    action::String,
)
    # Create relative conditions
    # Consistent (x, y) convention: x is horizontal (column), y is vertical (row)
    # When c1 is above c3, c1.y < c3.y, and y_offset should be negative
    # When c2 is to the left of c3, c2.x < c3.x, and x_offset should be negative
    x1_offset = c1.x - c3.x
    y1_offset = c1.y - c3.y
    x2_offset = c2.x - c3.x
    y2_offset = c2.y - c3.y

    cond1 = RelativeCondition(x1_offset, y1_offset, c1.item)
    cond2 = RelativeCondition(x2_offset, y2_offset, c2.item)

    # Convert state values to float32
    current_values = ValueTuple(convert.(Float32, copy(agent_state.values)))

    # Create expression string for debugging
    expr = "if $(c1.item) at ($(x1_offset),$(y1_offset)) and $(c2.item) at ($(x2_offset),$(y2_offset)) then $(c3.item)"

    # Create precondition
    precondition = Precondition(action, current_values, [cond1, cond2], expr)

    # Estimate reward change (this would need to be learned over time)
    reward_change = 0.0f0
    if c3.item == "goal"
        reward_change = 1.0f0
    elseif c3.item == "lava"
        reward_change = -1.0f0
    end

    # Create consequence - assume no change in values for now
    consequence = Consequence(
        reward_change,
        current_values,  # Same values until we observe changes
        c3.item,
        ValueTuple([0.0f0]),  # Would be updated with actual observed changes
        c3,
    )

    # Create rule
    Rule(precondition, consequence, 0.0f0, 0.0f0, 0.0f0, 0.0f0)
end

"""
    new_hypotheses(agent_state::NaceState, c3::Cell)

Generate new hypotheses for a given cell by looking at its neighborhood.

# Arguments

  - agent_state: Current agent state
  - c3: the cell to be used for the consequence
"""
function new_hypotheses(agent_state::NaceState, c3::Cell)
    percv_ext = agent_state.memory.episodic_current
    previous_externals = agent_state.memory.episodic_antecedant
    action = agent_state.memory.act_ante
    new_rules = Set{Rule}()

    # Check if we have the required board states
    if isempty(previous_externals.board) || isempty(percv_ext.board)
        @warn "Missing board in externals"
        return new_rules
    end

    board_ante = previous_externals.board
    board = percv_ext.board
    height, width = size(board)

    @debug "Generating hypotheses" cell_position = (c3.x, c3.y) item = c3.item action =
        action

    # Skip generating rules for persistently unseen cells
    if c3.item == "unseen" && board_ante[c3.x, c3.y].item == "unseen"
        @debug "Skipping persistently unseen cell"
        return new_rules
    end

    # Look at cells in a radius around the target cell
    radius = 1
    seen_combinations = Set{Tuple{String,String,String}}()  # Track unique item combinations

    # Get agent position (usually in the center)
    agent_y, agent_x = div(height, 2), div(width, 2)

    # Prioritize creating rules for cells that changed
    cell_changed = board_ante[c3.x, c3.y].item != c3.item

    # If the cell is in front of the agent, it's more important
    # Determine direction vector based on agent's orientation
    direction = percv_ext.direction
    dx, dy = 0, 0
    if direction == 0  # Right
        dx, dy = 1, 0
    elseif direction == 1  # Down
        dx, dy = 0, 1
    elseif direction == 2  # Left
        dx, dy = -1, 0
    elseif direction == 3  # Up
        dx, dy = 0, -1
    end

    # Check if this cell is in front of the agent
    in_front = (c3.x == agent_x + dx && c3.y == agent_y + dy)

    # Create more rules for important cells (changed or in front)
    max_rules = if cell_changed && in_front
        10  # More rules for important cells
    elseif cell_changed || in_front
        5   # Medium number for somewhat important cells
    else
        3   # Few rules for regular cells
    end

    rule_count = 0
    for i ∈ max(1, c3.y - radius):min(height, c3.y + radius)
        for j ∈ max(1, c3.x - radius):min(width, c3.x + radius)
            # Skip the cell itself
            (i == c3.y && j == c3.x) && continue

            # First precondition cell
            c1 = board_ante[i, j]

            # Skip if first cell is unseen
            c1.item == "unseen" && continue

            # Look for a second cell in the radius
            for k ∈ max(1, c3.y - radius):min(height, c3.y + radius)
                for l ∈ max(1, c3.x - radius):min(width, c3.x + radius)
                    # Skip the first cell and the target cell
                    (k == i && l == j) && continue
                    (k == c3.y && l == c3.x) && continue

                    # Early exit if we've generated enough rules for this cell
                    rule_count >= max_rules && return new_rules

                    c2 = board_ante[k, l]

                    # Skip if second cell is unseen
                    c2.item == "unseen" && continue

                    # Skip if we've already seen this combination of items
                    item_combo = (c1.item, c2.item, c3.item)
                    if item_combo in seen_combinations
                        continue
                    end
                    push!(seen_combinations, item_combo)

                    @debug "Considering cells" cell1_pos = (j, i) cell1_item = c1.item cell2_pos =
                        (l, k) cell2_item = c2.item

                    # Only generate rules for meaningful state changes
                    if c3.item != board_ante[c3.y, c3.x].item ||
                       (c3.item != c1.item && c3.item != c2.item)
                        # Create a rule linking these cells - updated to use new make_rule signature
                        rule = make_rule(agent_state, :BOARD, c3, c1, c2, action)

                        # Prioritize rules involving lava, goals, or walls
                        # important_item = any(item -> item in ["lava", "goal", "wall"], 
                        #                     [c1.item, c2.item, c3.item])

                        # Check if the rule is valid before adding it
                        if is_valid_rule(rule, agent_state.rules)
                            # If this is an important rule, give it initial positive evidence
                            # if important_item
                            #     rule.evidence_pos += 0.1f0
                            # end

                            push!(new_rules, rule)
                            rule_count += 1

                            @debug "Generated valid rule" precondition =
                                rule.precondition.expr consequence =
                                rule.consequence.cell_value
                        else
                            # Even if the rule is invalid, we should still track it as a negative rule
                            push!(new_rules, rule)
                            # Mark it as a negative rule with evidence
                            # FIXME: this is not what negative rules are.
                            # rule.evidence_neg += 0.2f0

                            @debug "Generated invalid rule as negative rule" precondition =
                                rule.precondition.expr consequence =
                                rule.consequence.cell_value
                        end
                    end
                end
            end
        end
    end

    @info "Rule generation complete" valid_rules = length(new_rules)
    return new_rules
end

function cycle(state::NaceState)::NaceState
    # Initialize empty sets for when we can't update rules
    focus = Set{Cell}()
    new_rules = Set{Rule}()
    new_negative_rules = Set{Rule}()
    M_change = Set{Cell}()
    M_prediction_mismatched = Set{Cell}()

    @info "==== CYCLE BEGIN (t=$(state.t)) ===="
    @info "Observer: Starting observation" focus_size = length(state.focus) rules_size =
        length(state.rules)

    # Create a previous state for comparison
    previous_state = NaceState(
        state.t - 1,
        state.focus,
        state.rules,
        state.values,
        Memory(
            state.memory.episodic_antecedant,
            EpisodicMemory(Matrix{Cell}(undef, 0, 0), Direction(0)),
            "",
        ),
    )

    # Update global bird view map with current perception
    # if haskey(state.memory.episodic_current, :BOARD)
    #     update_bird_view(state)
    #     @debug "Updated global bird view map"
    # end

    # Update rule evidence based on current observations
    @debug "Previous state exists, calculating changes..."
    M_change, M_observation_mismatched, M_prediction_mismatched =
        calculate_sets(previous_state, state)

    # Add cells around the agent's current position to focus
    board = state.memory.episodic_current.board
    height, width = size(board)
    # Find agent position (usually in the center of view)
    agent_x, agent_y = div(height, 2), div(width, 2)
    radius = 1

    # Add cells around agent to focus
    for i ∈ max(1, agent_x - radius):min(height, agent_x + radius)
        for j ∈ max(1, agent_y - radius):min(width, agent_y + radius)
            push!(M_change, board[j, i])
        end
    end
    @debug "Added cells around agent to focus" count = length(M_change)

    rule_memory = RuleMemory(state.rules)
    update_rule_memory(rule_memory)

    update_rule_evidence(
        rule_memory,
        M_change,
        M_observation_mismatched,
        M_prediction_mismatched,
    )

    @info "Observer: Updated bird view map" changes = length(M_change) prediction_mismatches =
        length(M_prediction_mismatched)

    # Predict the next state
    new_world = predict(state, 7, 7)

    # Hypothesize new rules
    try
        @debug "Attempting to hypothesize new rules..."
        focus, rule_evidence, new_rules, new_negative_rules = hypothesize(state)
        @info "Hypothesizer: Created rules" new_rules = length(new_rules) negative_rules =
            length(new_negative_rules)

        # Log some sample rules if available
        if !isempty(new_rules)
            sample_rules = collect(new_rules)[1:min(3, length(new_rules))]
            for (i, rule) ∈ enumerate(sample_rules)
                @info "Hypothesizer: Sample rule $i" rule = rule.precondition.expr consequence =
                    rule.consequence.cell_value
            end
        end

        # Log some sample negative rules if available
        if !isempty(new_negative_rules)
            sample_neg_rules =
                collect(new_negative_rules)[1:min(3, length(new_negative_rules))]
            for (i, rule) ∈ enumerate(sample_neg_rules)
                @info "Hypothesizer: Sample negative rule $i" rule = rule.precondition.expr consequence =
                    rule.consequence.cell_value
            end
        end
    catch e
        @error "Failed to hypothesize" exception = (e, catch_backtrace())
        # If hypothesizing fails, keep existing focus and no new rules
        focus = state.focus
        new_rules = Set{Rule}()
        new_negative_rules = Set{Rule}()
    end

    # Update focus based on prediction mismatches and changes
    if !isempty(state.memory.episodic_antecedant)
        focus = union(
            Set{Cell}(filter(x -> x isa Cell, state.focus)),
            M_change,
            M_prediction_mismatched,
        )
        @debug "Updated focus" size = length(focus)
    end

    # Plan the next actions using the improved planning system
    planned_actions, score, revisit_actions, age =
        plan(state, keys(ACTION_TO_IDX), 100, 2000, nothing)

    # Determine the next action, preferring planned actions over random ones
    action = if !isempty(planned_actions)
        @info "Planner: Using planned action sequence" actions =
            planned_actions[1:min(3, length(planned_actions))] score = score
        planned_actions[1]
    else
        # If no planned actions, use the action from the best rule
        best_rule = max_truth_exp(state.rules)
        if !isnothing(best_rule)
            @info "Planner: Using best rule action" action = best_rule.precondition.action score = truthexp(best_rule)
            best_rule.precondition.action
        else
            # Fallback to random action if no good rules exist
            # Only use meaningful actions (no "Unused")
            valid_actions =
                filter(a -> !startswith(a, "Unused"), collect(values(IDX_TO_ACTION)))
            random_action = rand(valid_actions)
            @info "Planner: Using random action" action = random_action
            random_action
        end
    end

    # Store the rules used for prediction
    rules_used_for_prediction = Set{Rule}()

    # Add strong negative evidence to rules that contradict observations
    for rule ∈ state.rules
        # If rule predicts something that didn't happen
        if rule.precondition.action == state.memory.act_ante &&
           rule.evidence_neg > rule.evidence_pos
            # Increase negative evidence for consistently wrong predictions
            rule.evidence_neg += 0.2f0
            @debug "Adding negative evidence to contradicting rule" rule =
                rule.precondition.expr
        end
    end

    # Update rule scores based on the chosen action
    for rule ∈ state.rules
        if rule.precondition.action == action
            rule.score += 0.1f0  # Small positive reinforcement for chosen action
            if truthexp(rule) > 0.6f0  # Only use high confidence rules for prediction
                push!(rules_used_for_prediction, rule)
            end
        end
    end

    @info "Predictor: Predicting next state based on action" action = action rules_used =
        length(rules_used_for_prediction)

    # If we have rules used for prediction, log a sample
    if !isempty(rules_used_for_prediction)
        sample_pred_rules =
            collect(rules_used_for_prediction)[1:min(2, length(rules_used_for_prediction))]
        for (i, rule) ∈ enumerate(sample_pred_rules)
            @info "Predictor: Using rule $i" rule = rule.precondition.expr consequence =
                rule.consequence.cell_value confidence = truthexp(rule)
        end
    end

    # Create new context with updated world state
    new_memory = Memory(new_world, state.memory.episodic_current, action)

    # Incorporate negative rules into the agent's knowledge
    # We'll use them to directly contradict positive rules and adjust their evidence
    for neg_rule ∈ new_negative_rules
        for existing_rule ∈ state.rules
            if cond_match(neg_rule.precondition, existing_rule.precondition) &&
               neg_rule.consequence.cell_value != existing_rule.consequence.cell_value
                # Add negative evidence to the existing rule
                existing_rule.evidence_neg += 0.5f0
                @debug "Applied negative rule to existing rule" existing_rule =
                    existing_rule.precondition.expr
            end
        end
    end

    # Periodically write rules to file for analysis
    if state.t % 10 == 0
        write_rules_to_file(state.rules, "rules_output.txt")
    end

    @info "==== CYCLE END (t=$(state.t)) ===="

    # Return the updated state
    return NaceState(
        state.t + 1,
        focus,
        union(state.rules, new_rules),  # Don't add negative rules directly, just use them to adjust positive rules
        state.values,
        new_memory,
    )
end

"""
    predict(state::NaceState, grid_width::Int, grid_height::Int)

Apply rules to predict the future world state. Returns a copy of the previous external state
if there are no rules or if the state is empty.
"""
function predict(state::NaceState, grid_width::Int, grid_height::Int)
    # If we have no previous state or rules, just return a copy of the current state
    if isempty(state.rules)
        return deepcopy(state.memory.episodic_current)
    end

    # Check if BOARD exists in the dictionaries
    if !haskey(state.memory.episodic_antecedant, :BOARD) ||
       !haskey(state.memory.episodic_current, :BOARD)
        return deepcopy(state.memory.episodic_current)
    end

    per_ext_post = deepcopy(state.memory.episodic_current)
    used_rules_sumscore = 0.0f0
    used_rules_amount = 0

    # Get the best rules for prediction - only use rules with positive evidence
    best_rules = choose_rules(state.rules)

    # If no good rules, try using all rules with usable truth expectation
    if isempty(best_rules)
        best_rules = filter(r -> truthexp(r) > 0.5f0, state.rules)
    end

    # If still no rules, just return the existing state
    isempty(best_rules) && return per_ext_post

    # Helper function to check if a rule is applicable at a specific cell
    function is_rule_applicable(rule::Rule, x::Int, y::Int, board)
        # Skip if action doesn't match
        if rule.precondition.action != state.memory.act_ante
            return false
        end

        # Check each condition in the rule
        for condition ∈ rule.precondition.conditions
            # Calculate absolute coordinates
            abs_x = x + condition.x_offset
            abs_y = y + condition.y_offset

            # Skip if coordinates are out of bounds
            if abs_y < 1 || abs_y > size(board, 1) || abs_x < 1 || abs_x > size(board, 2)
                return false
            end

            # Skip if cell value doesn't match condition
            if board[abs_y, abs_x].item != condition.value
                return false
            end
        end

        # All conditions matched
        return true
    end

    # Track which cells were predicted with which rules
    predictions = Dict{Tuple{Int,Int},Vector{Tuple{Rule,Float32}}}()

    # First pass: gather all possible predictions for each cell
    board = state.memory.episodic_current.board
    for y ∈ 1:grid_height, x ∈ 1:grid_width
        # Skip if coordinates are out of bounds
        if y > size(board, 1) || x > size(board, 2)
            continue
        end

        predictions[(y, x)] = Tuple{Rule,Float32}[]

        for rule ∈ best_rules
            # Check if rule is applicable at this cell
            if is_rule_applicable(rule, x, y, board)
                # Calculate confidence score
                confidence = truthexp(rule)

                # Only use rules with reasonable confidence
                if confidence > 0.5f0
                    push!(predictions[(y, x)], (rule, confidence))
                end
            end
        end

        # Sort predictions by confidence
        sort!(predictions[(y, x)], by=p -> -p[2])
    end

    # Second pass: apply the best predictions
    for (pos, preds) ∈ predictions
        if isempty(preds)
            continue
        end

        y, x = pos

        # Use the highest confidence prediction
        best_rule, confidence = preds[1]

        try
            # Update the cell with the predicted value
            per_ext_post.board[y, x] = Cell(x, y, best_rule.consequence.cell_value)
            used_rules_sumscore += best_rule.score
            used_rules_amount += 1
            @debug "Applied prediction rule" position = (x, y) rule =
                best_rule.precondition.expr confidence = confidence
        catch e
            @debug "Failed to apply rule at position" position = (x, y) error = e
            continue
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
A rule is valid if:

 1. It doesn't conflict with existing rules
 2. It represents a logical state transition
 3. It doesn't predict the same state for all cells
"""
function is_valid_rule(rule::Rule, existing_rules::Set{Rule})
    # Check for conflicts with existing rules
    if conflicting_rule_exists(rule, existing_rules)
        @debug "Rule conflicts with existing rule" rule = rule.precondition.expr
        return false
    end

    # Check if the rule represents a logical state transition
    # Some items can't change (like walls and lava)
    immutable_items = ["wall", "lava", "goal"]

    # Get cells from the rule
    target_item = rule.consequence.cell_value

    # Rules that predict walls/lava/goals appearing or disappearing are invalid
    # Need to check if any of the precondition cells are these immutable items
    is_immutable_change = false
    for condition ∈ rule.precondition.conditions
        if condition.x_offset == 0 && condition.y_offset == 0
            # This condition is for the target cell itself
            if condition.value != target_item &&
               (condition.value in immutable_items || target_item in immutable_items)
                is_immutable_change = true
                break
            end
        end
    end

    if is_immutable_change
        @debug "Rule predicts invalid state transition" to = target_item
        return false
    end

    # Don't allow rules that predict the same state for everything
    # (e.g., "if empty and empty then empty" is not useful)
    if length(rule.precondition.conditions) >= 2 &&
       all(
           c -> c.value == rule.precondition.conditions[1].value,
           rule.precondition.conditions,
       ) &&
       rule.precondition.conditions[1].value == target_item
        @debug "Rule predicts same state for everything" state = target_item
        return false
    end

    # Rules should be directional - the action should matter
    if isempty(rule.precondition.action)
        @debug "Rule has no action"
        return false
    end

    # Invisible cells need special handling
    # Don't allow rules that predict unseen cells unless it's a visibility transition
    if target_item == "unseen" &&
       !any(c -> c.value == "unseen", rule.precondition.conditions)
        @debug "Invalid rule predicting unseen cells"
        return false
    end

    # Special handling for actions
    # If the action is movement, make sure the rule makes sense spatially
    if rule.precondition.action == "Move forward"
        # For movement rules, check that conditions have reasonable spatial relationships
        # This is a simplification that could be improved
        has_adjacent_condition = false
        for condition ∈ rule.precondition.conditions
            # Check if any condition is adjacent to target (offset of 1 in any direction)
            if (abs(condition.x_offset) <= 1 && abs(condition.y_offset) <= 1) &&
               !(condition.x_offset == 0 && condition.y_offset == 0)
                has_adjacent_condition = true
                break
            end
        end

        if !has_adjacent_condition
            @debug "Movement rule with no adjacent cells"
            return false
        end
    end

    return true
end

"""
    conflicting_rule_exists(rule::Rule, rules::Set{Rule})

Check if there exists a conflicting rule in the set.
A rule conflicts if it has the same preconditions and action but predicts a different consequence.
"""
function conflicting_rule_exists(rule::Rule, rules::Set{Rule})
    for existing_rule ∈ rules
        # Check if actions match
        if rule.precondition.action != existing_rule.precondition.action
            continue
        end

        # Check if conditions match - count matches
        if length(rule.precondition.conditions) !=
           length(existing_rule.precondition.conditions)
            continue
        end

        # Check each condition - need to match all
        conditions_match = true
        for cond1 ∈ rule.precondition.conditions
            condition_found = false
            for cond2 ∈ existing_rule.precondition.conditions
                if cond1.x_offset == cond2.x_offset &&
                   cond1.y_offset == cond2.y_offset &&
                   cond1.value == cond2.value
                    condition_found = true
                    break
                end
            end

            if !condition_found
                conditions_match = false
                break
            end
        end

        # If all conditions match but consequence differs, it's a conflict
        if conditions_match &&
           rule.consequence.cell_value != existing_rule.consequence.cell_value
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

    # Check if we have access to value information
    has_values = !isempty(state.values)

    # Keep track of visited states to avoid cycles
    visited_states = Set{String}()

    # Helper function to create a hash for a state
    function state_hash(state::NaceState)
        if haskey(state.memory.episodic_current, :BOARD)
            board = state.memory.episodic_current.board
            # Create a string representation of the board
            return join([cell.item for cell ∈ vec(board)], "")
        end
        return "no_board"
    end

    while !isempty(queue) && length(queue) < max_queue_len
        current_state, action_seq, current_score = popfirst!(queue)

        # If we've reached max depth, skip this branch
        length(action_seq) >= max_depth && continue

        # Create a hash for this state
        state_key = state_hash(current_state) * join(action_seq, "")

        # Skip if we've visited this state already
        if state_key in visited_states
            continue
        end
        push!(visited_states, state_key)

        # Check for goal states - add a bonus for reaching a goal
        goal_bonus = 0.0f0
        if haskey(current_state.memory.episodic_current, :BOARD)
            board = current_state.memory.episodic_current.board
            for cell ∈ vec(board)
                if cell.item == "goal"
                    goal_bonus += 10.0f0  # Big bonus for finding a goal
                    @debug "Found goal in planning!" position = (cell.x, cell.y)
                elseif cell.item == "lava"
                    goal_bonus -= 5.0f0  # Penalty for being near lava
                    @debug "Found lava in planning!" position = (cell.x, cell.y)
                end
            end
        end

        # Try each possible action
        for action ∈ actions
            # Skip action sequences with repeated actions
            if !isempty(action_seq) &&
               last(action_seq) == string(action) &&
               action != "Move forward"
                continue  # Skip repeated turning actions
            end

            # Extra weight for forward movement to encourage exploration
            movement_bonus = action == "Move forward" ? 0.2f0 : 0.0f0

            # Small penalty for repeated turns to discourage spinning
            turn_penalty = action ∈ ["Turn left", "Turn right"] ? 0.1f0 : 0.0f0

            # Create a copy of the state and apply the action
            next_state = deepcopy(current_state)
            next_state.memory.act_ante = string(action)

            # Predict the next state
            predicted_world = predict(next_state, 7, 7)  # Using fixed size for now
            next_state.memory.episodic_current = predicted_world

            # Calculate score for this state
            # Base score is the state value
            base_score = state_value(next_state)

            # Add rewards from values (if available)
            value_score = 0.0f0
            if has_values && !isempty(next_state.values)
                value_score = sum(next_state.values) / length(next_state.values)
            end

            # Combine scores with appropriate weights
            total_score =
                current_score +
                (0.5f0 * base_score) +
                (1.0f0 * value_score) +
                goal_bonus +
                movement_bonus - turn_penalty

            # Update best score if this is better
            if total_score > best_score
                best_score = total_score
                best_actions = vcat(action_seq, [string(action)])
                @debug "New best plan" actions = best_actions score = best_score
            end

            # Add to queue if not at max depth
            if length(action_seq) < max_depth - 1
                push!(queue, (next_state, vcat(action_seq, [string(action)]), total_score))
            end
        end

        # Sort the queue by score (best first)
        sort!(queue, by=x -> -x[3])

        # Trim queue if it gets too large
        if length(queue) > max_queue_len ÷ 2
            queue = queue[1:max_queue_len÷2]
        end
    end

    return best_actions, best_score
end

"""
    write_rules_to_file(rules::Set{Rule}, filename::String; append::Bool=false)

Write the current set of rules to a file for analysis. If append is true, append to existing file.
"""
function write_rules_to_file(rules::Set{Rule}, filename::String; append::Bool=false)
    mode = append ? "a" : "w"
    open(filename, mode) do io
        if !append
            println(io, "Total rules: $(length(rules))")
            println(io, "===================")
        end
        
        # Add timestamp
        println(io, "\nTimestamp: $(now())")
        println(io, "Current rule count: $(length(rules))")
        println(io, "-------------------")

        # Group rules by action
        action_groups = Dict{String,Vector{Rule}}()
        for rule ∈ rules
            action = rule.precondition.action
            if !haskey(action_groups, action)
                action_groups[action] = Vector{Rule}()
            end
            push!(action_groups[action], rule)
        end

        # Print rules by action groups
        for (action, group_rules) ∈ action_groups
            println(io, "Action: $action ($(length(group_rules)) rules)")
            println(io, "-------------------")

            # Sort rules by evidence
            sort!(group_rules, by=r -> -(r.evidence_pos - r.evidence_neg))

            for (i, rule) ∈ enumerate(group_rules)
                println(io, "[$i] $rule")
                println(io, "Truth expectation: $(truthexp(rule))")
                println(io, "-------------------")
            end
            println(io)
        end
    end
    @info "Rules written to file" filename = filename rule_count = length(rules)
end

"""
    update_bird_view(state::NaceState)

Update the global bird's eye view map based on the current perception.
This maintains a persistent global map of the environment by combining observations.
"""
function update_bird_view(state::NaceState)
    board = state.memory.episodic_current.board

    # Get or create global map
    # if !haskey(state.memory.episodic_current, :GLOBAL_MAP)
    #     # Initialize global map with a reasonable size (larger than local view)
    #     global_height, global_width = 20, 20
    #     state.memory.episodic_current[:GLOBAL_MAP] =
    #         Matrix{Cell}([Cell(j, i, "unseen") for i ∈ 1:global_height, j ∈ 1:global_width])
    # end

    # global_map = state.memory.episodic_current[:GLOBAL_MAP]

    # Get agent position and direction
    # height, width = size(board)
    # agent_y, agent_x = div(height, 2), div(width, 2)
    direction = state.memory.episodic_current.direction

    # Calculate global center position (where agent is located)
    # Assume global map center is at (10, 10) for simplicity
    # global_center_y, global_center_x = 10, 10

    # Update global map based on current perception
    for I ∈ CartesianIndices(board)
        # Calculate relative position to agent
        rel_y = I[1] - agent_y
        rel_x = I[2] - agent_x

        # Convert to global coordinates
        global_y = global_center_y + rel_y
        global_x = global_center_x + rel_x

        # Check if global coordinates are valid
        if 1 <= global_y <= size(global_map, 1) && 1 <= global_x <= size(global_map, 2)
            # Only update if cell is visible (not "unseen")
            if board[I].item != "unseen"
                global_map[global_y, global_x] = Cell(global_x, global_y, board[I].item)
            end
        end
    end

    # Log the update
    @debug "Updated global bird view map" size = size(global_map)

    return state.memory.episodic_current
end
