export Rule, rule_applicable, Cell, State, rule_ratio, cell_value, state_value, truthexp

struct Cell
    x::Int
    y::Int
    item::String
end

struct AgentContext
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
    for rule ∈ rules
        c1 = rule.precondition.cell1
        c2 = rule.precondition.cell2
        c3 = rule.consequence.cell
        if Set([c1, c2, c3]) ⊆ m
            rule.evidence_pos += learning_rate
        end
        if c3 ∈ M_prediction_mismatched
            rule.evidence_neg += learning_rate
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

    prev_board = previous_state.context.perceived_externals[:BOARD]
    curr_board = current_state.context.perceived_externals[:BOARD]

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
    precondition_str =
        replace(rule.precondition.expr, r"VALUES\s*==\s*\[(.*?)\]" => s"VALUES =\n\1")
    precondition_str = replace(precondition_str, r"DIR\s*==\s*(\d+)" => s"DIR =\n\1")
    precondition_str = format_2d_array(replace(
        precondition_str,
        r"BOARD\s*==\s*\[(.*?)\]" => s"BOARD =\n\1",
    ))

    consequence_str = string(rule.consequence)
    consequence_str = replace(consequence_str, r"VALUES\s*=\s*\[(.*?)\]" => s"VALUES =\n\1")
    consequence_str = replace(consequence_str, r"DIR\s*=\s*(\d+)" => s"DIR =\n\1")
    consequence_str = format_2d_array(replace(
        consequence_str,
        r"BOARD\s*=\s*\[(.*?)\]" => s"BOARD =\n\1",
    ))

    print(
        io,
        "Rule[\nPrecondition:\n$precondition_str,\n\nConsequence:\n$consequence_str,\nScore: $(rule.score)\n]",
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

Base.show(io::IO, cond::Precondition) = print(io, "Condition(Expression: $(cond.expr))")

# Added definition for Consequence
function Base.show(io::IO, c::Consequence)
    print(io, "Consequence(cell=$(c.cell), reward=$(c.reward))")
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
    state_value(s::State)

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
