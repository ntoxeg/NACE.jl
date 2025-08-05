# Post-Mortem: NACE Performance Issues

## Problem

The NACE agent was experiencing severe performance issues, causing it to hang indefinitely after just a few cycles. Initial investigation revealed two main problems:

1.  **Combinatorial Explosion:** The `hypothesize` function was generating an excessive number of rules, leading to a combinatorial explosion that quickly overwhelmed the system.
2.  **Unbounded Focus Growth:** The agent's `focus` set was growing without bounds, causing the `hypothesize` function to do more and more work on each cycle.

A number of `MethodError` exceptions were also encountered related to the introduction of a new field to the `NaceState` struct.

## What Didn't Work

*   **Limiting Rules at the End of `hypothesize`:** My initial attempt to fix the combinatorial explosion was to limit the number of rules returned by the `hypothesize` function. This was ineffective because the expensive computation had already been done.
*   **Incorrectly Clearing the `focus` Set:** My first attempt to clear the `focus` set was flawed. I was clearing the set before passing it to the `previous_state`, which was incorrect.
*   **Running one-line Julia commands from the shell:** Repeatedly failed due to quoting issues.

## What Worked

The following changes successfully resolved the performance issues and bugs:

*   **Added `Dates` dependency:** Fixed the `UndefVarError: now not defined` error by adding `using Dates` to `src/base.jl` and adding `Dates` as a project dependency.
*   **Limited Rule Generation (Correctly):** I introduced a `max_new_rules_per_cycle` parameter and used it to break the rule generation loop in `hypothesize` once the limit was reached. This prevented the combinatorial explosion from happening in the first place.
*   **Optimized Precondition Sampling:** I replaced the brute-force nested loops in `new_hypotheses` with random sampling of neighbors. This drastically reduced the amount of computation required to generate new rules.
*   **Cleared the `focus` Set (Correctly):** I modified the `cycle` function to clear the `focus` set *after* it had been used to create the `previous_state`. This prevents the unbounded growth of the `focus` set and keeps the agent focused on immediate changes in its environment.
*   **Fixed all `NaceState` constructor calls:** Updated all `NaceState` constructor calls to include the new `max_new_rules_per_cycle` field.

With these changes in place, the agent is now able to run for many cycles without hanging, and all tests are passing.
