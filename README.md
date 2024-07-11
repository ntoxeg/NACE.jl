# NACE.jl
> An implementation of [Non-Axiomatic Causal Explorer](https://github.com/patham9/NACE) in Julia

[![Build Status](https://github.com/ntoxeg/NACE.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ntoxeg/NACE.jl/actions/workflows/CI.yml?query=branch%3Amain)

## TODO
- [ ] Make experiment no. 2 run.
    - [ ] `hypothesize` and other unimplemented functions need to reflect the original logic flow diagram. Multiple functionalities condensed in the current functions need to be split out into those functions.
        - [ ] `hypothesize`
        - [ ] `prediction_errors`
        - [ ] `new_hypotheses`
        - [ ] `verify_hypothesis`
        - [ ] `max_truth_exp`
        - [ ] `best_hypothesis`
        - [ ] `highest_reward`
        - [ ] `weakest_hypothesis`
        - [ ] `oldest_observed`
- [ ] Remove old code (everything currently outside `NACE.jl`), preserving useful parts.
- [ ] Lastly, refactor `NACE.jl` into multiple files to have a clean library structure.
