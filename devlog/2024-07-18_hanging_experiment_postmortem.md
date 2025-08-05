# Postmortem: Unresolved Hanging Issue in `experiment02.jl`

**Date:** 2024-07-18

**Author:** AI Assistant

## Summary

This document outlines the debugging process for an issue where `experiments/experiment02.jl` would hang indefinitely after a certain number of steps. The initial problem was that the agent was not generating any rules. While this was fixed, it led to a performance bottleneck that could not be resolved, and the task was concluded as a failure.

## Initial Problem

The agent was not generating any rules. The `focus` set, which is crucial for hypothesis generation, was not being updated correctly, leading to an empty set of new rules in each cycle.

## Debugging Steps and Rationale

1.  **Initial Investigation:**
    *   **Action:** Analyzed the core logic in `src/base.jl`, specifically the `cycle`, `hypothesize`, and `new_hypotheses` functions.
    *   **Rationale:** The problem was clearly in the rule generation pipeline. The initial hypothesis was that the conditions for creating new rules were too strict.

2.  **Relaxing Rule Generation:**
    *   **Action:** Modified `new_hypotheses` to be less restrictive in how it selected cells and determined the number of rules to generate.
    *   **Rationale:** To increase the probability of rule generation by broadening the criteria.
    *   **Outcome:** This did not solve the issue. The `focus` set remained empty.

3.  **Refactoring Focus Set Management (Attempt 1):**
    *   **Action:** Corrected the logic in the `cycle` function for updating the `focus` set. The `focus` set was being re-initialized as empty in every cycle instead of being accumulated.
    *   **Rationale:** To ensure that the `focus` set would grow over time and provide input for the `hypothesize` function.
    *   **Outcome:** The agent started generating rules. However, this led to a new problem: the IDE crashed due to excessive memory usage as the `focus` set grew without bounds.

4.  **Implementing a "Forgetting" Mechanism (Attempt 1):**
    *   **Action:** A simple "forgetting" mechanism was added to cap the size of the `focus` set by randomly sampling from it when it exceeded a threshold.
    *   **Rationale:** To prevent uncontrolled memory growth.
    *   **Outcome:** This led to dependency issues (`Random` package not being in `Project.toml`) and persistent precompilation errors.

5.  **Refactoring Focus Set Management (Attempt 2 - Inspired by Original NACE):**
    *   **Action:** Inspected the Python implementation of NACE. It was discovered that the `FocusSet` was a dictionary (`Dict`) mapping item types (strings) to an integer "interestingness" score, not a set of `Cell` objects. The implementation was refactored to match this design.
    *   **Rationale:** To align with the original, working implementation and adopt a more sophisticated and potentially more memory-efficient way of managing focus.
    *   **Outcome:** The agent successfully generated rules without crashing the IDE immediately. However, the experiment would hang after a few dozen steps.

6.  **Addressing the Hanging Issue:**
    *   **Action:** Hypothesized that the `plan` function's breadth-first search was the bottleneck, as its complexity grows with the number of rules. The `max_depth` and `max_queue_len` parameters were significantly reduced to constrain the search space.
    *   **Rationale:** To reduce the computational load of the planning phase and prevent it from running indefinitely.
    *   **Outcome:** The experiment still hung, even with a drastically reduced search space for the planner.

## Conclusion

The primary issue of rule generation was solved by refactoring the focus set management to align with the original Python implementation. However, this introduced a performance bottleneck that causes the experiment to hang. Reducing the planner's search parameters did not resolve the hanging.

The root cause of the hanging is likely still within the `plan` function or a related part of the prediction logic that becomes computationally intractable as the number of rules increases. The task is being closed as a failure, as the hanging issue could not be resolved. Future work should focus on profiling the `cycle` function, especially the `plan` and `predict` calls, to identify the exact source of the performance bottleneck.
