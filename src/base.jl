export Rule, applicable, Cell, State, rule_ratio, cell_value, state_value, truthexp

struct Precondition
    cell1::Any
    cell2::Any
    agent_state::Any
    action::Any
end

struct Consequence
    cell::Any
    agent_state::Any
    reward::Any
end

struct Rule
    precondition::Precondition
    consequence::Consequence
    evidence_pos::Int32
    evidence_neg::Int32
    score::Float32
    acc_score::Float32
end

"""
    NaceState(t, focus, perceived_externals, per_ext_ante, act_ante, rules, values)

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
    rules::Set{Rule}
    values::Vector{Int}
end

function truthexp_with(cfun::Function, r::Rule)::AbstractFloat
    w = r.evidence_neg + r.evidence_pos
    f = r.evidence_pos / w
    c = cfun(w)
    f * c + 0.5 * (1 - c)
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
    rules = rulem.indeterminate_rules ∪ rulem.active_rules ∪ rulem.inactive_rules
    m = M_change ∪ M_observation_mismatched
    for rule ∈ rules
        c1 = rule.precondition.cell1
        c2 = rule.precondition.cell2
        c3 = rule.consequence.cell
        if Set([c1, c2, c3]) ⊆ m
            rule.evidence_pos += 1
        end
        if c3 ∈ M_prediction_mismatched
            rule.evidence_neg += 1
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

    prev_board = previous_state.perceived_externals[:BOARD]
    curr_board = current_state.perceived_externals[:BOARD]

    # Calculate changes between states
    for i ∈ 1:size(prev_board, 1), j ∈ 1:size(prev_board, 2)
        prev_cell = prev_board[i, j]
        curr_cell = curr_board[i, j]

        if prev_cell.item != curr_cell.item
            push!(M_change, curr_cell)
        end
    end

    # Calculate prediction mismatches
    predicted_board =
        predict(previous_state, size(prev_board, 1), size(prev_board, 2))[:BOARD]
    for i ∈ 1:size(curr_board, 1), j ∈ 1:size(curr_board, 2)
        pred_cell = predicted_board[i, j]
        curr_cell = curr_board[i, j]

        if pred_cell.item != curr_cell.item
            push!(M_prediction_mismatched, curr_cell)
        end
    end

    # Calculate observation mismatches
    for cell ∈ M_change
        if cell in M_prediction_mismatched
            push!(M_observation_mismatched, cell)
        end
    end

    return M_change, M_observation_mismatched, M_prediction_mismatched
end

function Base.show(io::IO, rule::Rule)
    precondition =
        replace(rule.precondition.expr, r"VALUES\s*==\s*\[(.*?)\]" => s"VALUES =\n\1")
    precondition = replace(precondition, r"DIR\s*==\s*(\d+)" => s"DIR =\n\1")
    precondition =
        format_2d_array(replace(precondition, r"BOARD\s*==\s*\[(.*?)\]" => s"BOARD =\n\1"))
    consequence = replace(rule.consequence, r"VALUES\s*=\s*\[(.*?)\]" => s"VALUES =\n\1")
    consequence = replace(consequence, r"DIR\s*=\s*(\d+)" => s"DIR =\n\1")
    consequence =
        format_2d_array(replace(consequence, r"BOARD\s*=\s*\[(.*?)\]" => s"BOARD =\n\1"))
    print(
        io,
        "Rule[\nPrecondition:\n$precondition,\n\nConsequence:\n$consequence,\nScore: $(rule.score)\n]",
    )
end

function format_rule_comp(key::AbstractString, comp::AbstractString)
    rows = split(comp, ";")
    array_rows = map(row -> split(strip(row)), rows)
    if key == "VALUES"
        convert_row_int(row) = map(el -> parse(Int32, String(el)), row)
        array_rows = map(row -> convert_row_int(row), array_rows)
        prefix = "$key = \n"
    end
    if key == "BOARD"
        convert_row_str(row) = map(el -> replace(el, "\"" => ""), row)
        array_rows = map(row -> convert_row_str(row), array_rows)
        prefix = "$key = \n"
    end
    if key == "DIR"
        convert_dir(row) = map(el -> parse(Int32, String(el)), row)
        array_rows = map(row -> convert_dir(row), array_rows)
        prefix = "$key = "
    end
    data = length(array_rows) > 1 ? stack(array_rows) : array_rows[1][1]
    prefix * repr("text/plain", data)
end

function format_2d_array(s::AbstractString)
    comps = split(s, "=")
    fmtstr = format_rule_comp(strip(comps[1]), comps[2])
    if length(comps) == 4
        fmtstr2 = format_rule_comp(strip(comps[3]), comps[4])
        return fmtstr, fmtstr2
    else
        return fmtstr
    end
end

Base.show(io::IO, cond::Precondition) = print(io, "Condition(Expression: $(cond.expr))")

struct Cell
    x::Int
    y::Int
    item::Any
end

# TODO: determine Condition structure
function cond_match(cond1::Precondition, cond2::Precondition)
    cond1.expr == cond2.expr
end

"""
    rule_ratio(c::Cell, r::Rule)

Calculate the match ratio of a rule for a given cell.
"""
function rule_ratio(c::Cell, r::Rule)
    length(filter(cx -> cond_match(r.precondition, cx), c.conds)) / length(c.conds)
end

"""
    cell_value(rs::Vector{Rule}, c::Cell)

Calculate the match value of a cell.

The match value of a cell is the maximum of rule match ratios, for all possible rules.
"""
function cell_value(rs::Set{Rule}, c::Cell)
    max(map(r -> rule_ratio(c, r), rs))
end

"""
    state_value(s::State)

Calculate the match value of a state.

The state match value is the maximum match value of all cells in the state.
"""
function state_value(s::NaceState)
    max(map(c -> cell_value(s.rules, c), s.perceived_externals[:BOARD]))
end

"""
    applicable(sv::Float64, rr::Float64)::Bool

Determine whether a rule is applicable based on its match ratio relative to the state value.
"""
function applicable(sv::Float64, rr::Float64)::Bool
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
