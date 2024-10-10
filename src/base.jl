export Rule, applicable, Cell, State, rule_ratio, cell_value, state_value

struct Condition
    expr::String
end

Base.show(io::IO, rule::Rule) = print(io, "Rule(Precondition: $(rule.precondition.expr), Consequence: $(rule.consequence), Score: $(rule.score))")

Base.show(io::IO, cond::Condition) = print(io, "Condition(Expression: $(cond.expr))")

struct Rule
    precondition::Condition
    consequence::String
    score::Float32
    acc_score::Float32
    # v_inventory
    # TODO: determine Rule structure
end

struct Cell
    x::Int
    y::Int
    conds::Set{Condition}
end

struct State
    cells::Set{Cell}
    rules::Set{Rule}
end

# TODO: determine Condition structure
function cond_match(cond1::Condition, cond2::Condition)
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
function state_value(s::State)
    max(map(c -> cell_value(s.rules, c), s.cells))
end

"""
    applicable(sv::Float64, rr::Float64)::Bool

Determine whether a rule is applicable based on its match ratio relative to the state value.
"""
function applicable(sv::Float64, rr::Float64)::Bool
    rr > 0.0 && rr == sv
end
