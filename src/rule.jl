export Rule, applicable

struct Rule
    precondition::String
    consequence::String
    score::Float32
    acc_score::Float32
    v_inventory
end

"""
    applicable(highscore::Float64, highesthighscore::Float64) :: Bool

Determine whether a rule is applicable based on its score relative to the highest score.
"""
function applicable(highscore::Float64, highest_highscore::Float64, rule::Rule)::Bool
    return highscore > 0.0 && highscore == highest_highscore
end
