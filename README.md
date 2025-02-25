# NACE.jl
> An implementation of [Non-Axiomatic Causal Explorer](https://github.com/patham9/NACE) in Julia

[![Build Status](https://github.com/ntoxeg/NACE.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ntoxeg/NACE.jl/actions/workflows/CI.yml?query=branch%3Amain)

## TODO
- [ x ] Make experiment no. 2 run.
    - [ x ] `hypothesize` and other unimplemented functions need to reflect the original logic flow diagram. Multiple functionalities condensed in the current functions need to be split out into those functions.
        - [ x ] `hypothesize`
        - [ x ] `new_hypotheses`
        - [ x ] `verify_hypothesis`
        - [ x ] `max_truth_exp`
        - [ x ] `best_hypothesis`
        - [ x ] `highest_reward`
        - [ x ] `weakest_hypothesis`
        - [ x ] `oldest_observed`
- [ x ] Remove old code (everything currently outside `NACE.jl`), preserving useful parts (mostly done).
- [ x ] Lastly, refactor `NACE.jl` into multiple files to have a clean library structure.
- [ ] Next on the agenda, make the agent actually learn something in the experiment no. 2.
  - [ x ] Score updates currently don't do anything, figure out why.
  - [ x ] Verify that the rule gen logic makes sense.
  - [ ] Verify the rest of the code.

## Install
I recommend using Poetry to make a Python virtual env.
Spawn the env's shell (`eval $(poetry env activate)` or `Invoke-Expression (poetry env activate)`) and run the REPL with `julia --project`, before installing dependencies (`] instantiate`) make sure
to execute `ENV["PYTHON"] = Sys.which("python")` -- that will set PyCall to use your environment's Python. You don't have to run julia from the activated environment after that.

Alternatively, you can run `poetry run julia --project -e 'ENV["PYTHON"] = Sys.which("python"); using Pkg; Pkg.instantiate()'` to do all of the above in one line.

## Run
In the REPL, `using NACE` should be enough, some things may be not exported (the API is not yet stable, needless to say) -- those you have to access under the packages namespace (`NACE`). There is a comment in NACE.jl that shows how to spawn a Gym environment in Julia.

There are experiment files that you can run with `julia --project experiments/<exp name>.jl`.

Under the hood, environments are provided by the Farama Foundation's Minigrid library, which
relies on the Gymnasium package, also maintained by the Foundation.
