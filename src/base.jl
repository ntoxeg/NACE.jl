export Rule, applicable, Cell, State, rule_ratio, cell_value, state_value

struct Condition
    expr::String
end

struct Rule
    precondition::Condition
    consequence::String
    score::Float32
    acc_score::Float32
    # v_inventory
    # TODO: determine Rule structure
end

function Base.show(io::IO, rule::Rule)
    precondition =
        replace(rule.precondition.expr, r"VALUES\s*==\s*\[(.*?)\]" => s"VALUES =\n\1")
    precondition = replace(precondition, r"DIR\s*==\s*\[(.*?)\]" => s"DIR =\n\1")
    precondition =
        format_2d_array(replace(precondition, r"BOARD\s*==\s*\[(.*?)\]" => s"BOARD =\n\1"))
    consequence = replace(rule.consequence, r"VALUES\s*=\s*\[(.*?)\]" => s"VALUES =\n\1")
    consequence = replace(consequence, r"DIR\s*=\s*\[(.*?)\]" => s"DIR =\n\1")
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
    end
    if key == "BOARD"
        convert_row_str(row) = map(el -> replace(el, "\"" => ""), row)
        array_rows = map(row -> convert_row_str(row), array_rows)
    end
    if key == "DIR"
        array_rows = [parse(Int32, String(array_rows[1]))]
    end
    data = length(array_rows) > 1 ? stack(array_rows) : array_rows[1]
    "$key = \n$(repr("text/plain", data))"
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

Base.show(io::IO, cond::Condition) = print(io, "Condition(Expression: $(cond.expr))")

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
