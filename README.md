# NACE.jl
> An implementation of [Non-Axiomatic Causal Explorer](https://github.com/patham9/NACE) in Julia

[![Build Status](https://github.com/ntoxeg/NACE.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ntoxeg/NACE.jl/actions/workflows/CI.yml?query=branch%3Amain)

## TODO
- [ x ] Make experiment no. 2 run.
    - [ x ] `hypothesize` and other unimplemented functions need to reflect the original logic flow diagram. Multiple functionalities condensed in the current functions need to be split out into those functions.
        - [ x ] `hypothesize`
        - [ x ] `new_hypotheses`
        - [ x ] `verify_hypothesis`
        - [ ] `max_truth_exp` (removed for now)
        - [ ] `best_hypothesis` (removed for now)
        - [ ] `highest_reward` (removed for now)
        - [ ] `weakest_hypothesis`(removed for now)
        - [ ] `oldest_observed` (removed for now)
- [ ] Remove old code (everything currently outside `NACE.jl`), preserving useful parts (mostly done).
- [ ] Lastly, refactor `NACE.jl` into multiple files to have a clean library structure.
- [ ] Next on the agenda, make the agent actually learn something in the experiment no. 2.
  - [ ] Score updates currently don't do anything, figure out why.
  - [ ] Verify that the rule gen logic makes sense.
  - [ ] Verify the rest of the code.

## Install
I recommend using Poetry to make a Python virtual env.
Spawn the env's shell and run the REPL with `julia --project`, before installing deps make sure
to execute `ENV["PYTHON"] = Sys.which("python")` -- that will set PyCall to use your env. You don't have to run julia from the virtual env after that as PyCall will be set to that env's interpreter.

In the REPL invoke `] instantiate` to install Julia deps.

## Run
In the REPL, `using NACE` should be enough, some things maybe not exported (the API is not yet stable, needless
to say) -- those you have to access under the packages namespace (`NACE`). There is a comment in NACE.jl that shows how to spawn a Gym env in Julia.

There are experiment files that you can run with `julia --project experiments/<exp name>.jl`.

Under the hood, environments are provided by the Farama Foundation's Minigrid library, which
relies on the Gymnasium package, also maintained by the Foundation.
