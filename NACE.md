# NACE Analysis

This document analyzes the original NACE agent, focusing on its focus set management and tree search algorithm, based on the Python implementation in the `reporef/NACE` directory.

## Focus Set Management

In the original Python implementation, the concept of a "focus set" is realized through a dictionary named `FocusSet`. This dictionary is used to track items of interest in the environment, guiding the agent's attention and rule-generation processes.

The `_Observe` function in `nace.py` is responsible for managing the `FocusSet`. It operates as follows:

1.  **Identifying Unique Items:** The agent counts the occurrences of each item in its current field of view. If an item appears only once (i.e., it's unique), it is added to the `FocusSet`.

2.  **Tracking Changes:** When a unique item changes, its corresponding value in the `FocusSet` dictionary is incremented. This mechanism allows the agent to prioritize elements that are not only rare but also dynamic.

The `_MatchHypotheses` function leverages the `FocusSet` to create an `AttendPositions` set. This set includes the coordinates of the focused items and their adjacent cells, effectively narrowing the scope of rule matching to the most relevant parts of the environment.

## Tree Search Algorithm

The NACE agent's planning capabilities are centered around the `_Plan` function in `nace.py`, which implements a breadth-first search (BFS) to explore possible action sequences. The search is not exhaustive but is instead constrained by two key parameters:

*   **`max_depth`**: This parameter limits how many steps into the future the planner will look. The default value is set to 100.
*   **`max_queue_len`**: This parameter restricts the number of states that can be held in the planning queue, preventing the search from becoming computationally prohibitive. The default value is 2000.

The `_Plan` function uses the learned rules to predict the consequences of actions, creating a search tree of possible futures. Each path in the tree is evaluated based on a scoring system that considers potential rewards and the reduction of uncertainty. This allows the agent to find a balance between exploiting its knowledge to achieve goals and exploring the environment to learn more.

## Symbol Reference

### `nace.py`

- `FocusSet` (dict): Tracks items of interest in the environment and counts their salience.
- `rules` (set): Current pool of learned causal rules.
- `negrules` (set): Rules that have gathered predominately negative evidence.
- `worldchange` (set): Helper set used while detecting changes in the grid world.
- `RuleEvidence` (dict): Maps each rule to a tuple `(wp, wn)` of positive/negative evidence counts.
- `observed_world` (list): Partial internal world model holding board layout, reward values, and timestamps.
- `nochange` (bool): Flag indicating that no observable change happened in the last step.

- `NACE_Cycle(...)`: Executes one observe-learn-plan-act iteration; orchestrates perception, rule update, planning and action selection.
- `NACE_Predict(...)`: Uses the rule set to simulate a forward step, returning the predicted world, AIRIS score, age metric, and value vector.
- `_Plan(...)`: Breadth-first forward search bounded by depth and queue length; returns best action sequence, its score, alternative revisit plan and age of oldest focus element.
- `_IsPresentlyObserved(...)`: Tests whether a cell was observed at the current time step.
- `_AddToAdjacentSet(...)`: Maintains groups of adjacent cells considered together when analysing changes.
- `_Observe(...)`: Generates positive/negative evidence and derives new or specialized rules from observed changes and prediction mismatches.
- `_MatchHypotheses(...)`: Scores every rule against possible spatial anchors, returning per-position best scores.
- `_RuleApplicable(...)`: Predicate that determines whether a rule should be applied given the scoring context.
- `NACE_PrintScore(...)`: Prints a human-friendly view of the prediction/goal score used during planning.

### `hypothesis.py`

- `Hypothesis_UseMovementOpAssumptions(...)`: Registers the symbolic names of movement operations and toggles symmetry assumptions.
- `Hypothesis_TruthValue(wpn)`: Converts evidence counters into `(frequency, confidence)`.
- `Hypothesis_TruthExpectation(tv)`: Expected truth value derived from frequency and confidence, used for ranking rules.
- `Hypothesis_Choice(...)`: Chooses the rule with the higher truth expectation when two rules clash.
- `Hypothesis_Contradicted(...)`: Records negative evidence for a rule; may demote it or move it to the negative set.
- `Hypothesis_Confirmed(...)`: Records positive evidence and, if appropriate, abduces rotated/operation-independent variants.
- `Hypothesis_ValidCondition(cond)`: Checks whether a precondition offset is within the allowed Moore-like neighbourhood.
- `Hypothesis_BestSelection(...)`: Prunes the rule set to those with sufficient truth expectation and removes inferior duplicates.
- `_OpRotate(op)`: Returns the operation rotated 90° clockwise.
- `_ConditionRotate(cond)`: Rotates the coordinate of a condition accordingly.
- `_Variants(...)`: Generates symmetry-based or operation-independent variants of a rule, aiding faster generalisation.
- `_AddEvidence(...)`: Maintains capped positive/negative evidence counts, enabling gradual forgetting in non-stationary worlds.

