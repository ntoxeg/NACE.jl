export Rule, applicable

struct Rule
    precondition::String
    consequence::String
    score::Float64
end

"""
    applicable(highscore::Float64, highesthighscore::Float64) :: Bool

Determine whether a rule is applicable based on its score relative to the highest score.
"""
function applicable(highscore::Float64, highesthighscore::Float64)::Bool
    return highscore > 0.0 && highscore == highesthighscore
end
