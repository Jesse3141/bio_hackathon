# Profile HMM with Insertion/Deletion States

**Date**: 2026-01-12
**Status**: Approved

## Overview

Extend the Full-Sequence HMM to include insertion and deletion states at every position, enabling the model to handle variable-length observations (missing or extra signal segments).

## State Architecture

### Non-cytosine positions (147 positions)

Each position `i` has 3 states:
- **M_i** (Match): Emits from Gaussian N(μ_i, σ_i) learned from data
- **I_i** (Insert): Emits from Uniform(data_min, data_max)
- **D_i** (Delete): Silent, no emission

### Cytosine positions (8 positions: 38, 50, 62, 74, 86, 98, 110, 122)

Each cytosine position has:
- **M_C_i**: Match for canonical C, emits N(μ_C, σ_C)
- **I_C_i**: Insert for C path, emits Uniform
- **M_5mC_i**: Match for 5mC, emits N(μ_5mC, σ_5mC)
- **I_5mC_i**: Insert for 5mC path, emits Uniform
- **M_5hmC_i**: Match for 5hmC (3-way only), emits N(μ_5hmC, σ_5hmC)
- **I_5hmC_i**: Insert for 5hmC path (3-way only), emits Uniform
- **D_i**: Single deletion state (shared across modification paths)

### State counts

| Mode | Non-cyt states | Cytosine states | Total |
|------|----------------|-----------------|-------|
| Binary | 147 × 3 = 441 | 8 × (2×2 + 1) = 40 | **481** |
| 3-way | 147 × 3 = 441 | 8 × (3×2 + 1) = 56 | **497** |

(Previously: 163 binary, 171 3-way)

## Transition Probabilities

### From Match states (M_i)

```
M_i → M_{i+1}   P = 0.85  (forward to next match)
M_i → I_i       P = 0.05  (enter insertion at current position)
M_i → D_{i+1}   P = 0.10  (delete next position)
```

### From Insertion states (I_i)

```
I_i → M_{i+1}   P = 0.70  (exit insertion, advance)
I_i → I_i       P = 0.30  (self-loop, absorb more extras)
```

### From Deletion states (D_i)

```
D_i → M_{i+1}   P = 0.70  (exit deletion, resume matching)
D_i → D_{i+1}   P = 0.30  (extend deletion, skip another)
```

### Fork transitions at cytosine positions

When entering a cytosine position from position `i-1`:

```
M_{i-1} → M_C_i      P = 0.85 / n_mods
M_{i-1} → M_5mC_i    P = 0.85 / n_mods
M_{i-1} → M_5hmC_i   P = 0.85 / n_mods  (3-way only)
M_{i-1} → D_i        P = 0.10           (delete this cytosine)
```

Within a fork path, insertions follow their modification:
```
M_C_i   → I_C_i      P = 0.05
I_C_i   → I_C_i      P = 0.30  (self-loop)
I_C_i   → M_{i+1}    P = 0.70  (exit to next position)
```

## Emission Distributions

### Match states

Gaussian distributions from data:
- **Non-cytosine M_i**: N(μ_i, σ_i) from control sample at position i
- **M_C_i**: N(μ_C, σ_C) from control sample at cytosine position
- **M_5mC_i**: N(μ_5mC, σ_5mC) from 5mC sample
- **M_5hmC_i**: N(μ_5hmC, σ_5hmC) from 5hmC sample

### Insertion states

Uniform distribution over observed data range:
```python
data_min = min(all_current_values)  # e.g., ~400 pA
data_max = max(all_current_values)  # e.g., ~1200 pA
I_emission = Uniform(data_min, data_max)
```

All insertion states share the same uniform distribution.

### Deletion states

Silent - no emission. Handled in scoring by allowing paths that don't consume observations.

## Classification Algorithm

Dynamic programming with I/D state handling:

```python
def score_modification(observations, mod):
    """
    DP over positions, tracking whether we're in M, I, or D.

    dp[obs_idx][pos][state] = log probability
    """
    T = len(observations)
    N = SEQUENCE_LENGTH  # 155

    for t, obs in enumerate(observations):
        for pos in range(N):
            # Match: consumes observation, advances position
            dp[t+1][pos+1]['M'] = dp[t][pos]['M'] + log_emit(obs, pos, mod)

            # Insert: consumes observation, stays at position
            dp[t+1][pos]['I'] = dp[t][pos]['I'] + log_uniform(obs)

            # Delete: NO observation consumed, advances position
            dp[t][pos+1]['D'] = dp[t][pos]['M'] + log(P_DELETE)

    return best_final_score

scores = {mod: score_modification(obs, mod) for mod in modifications}
prediction = argmax(scores)
```

Key: Deletion transitions don't increment observation index, only position index.

## Implementation Plan

### Files to modify

1. **`methylation_hmm/full_sequence_hmm.py`**
   - Update `_build_state_mapping()` for I/D states
   - Update `_build_distributions()` with Uniform for insertions
   - Update `_build_transition_matrix()` with I/D transitions
   - Replace `predict_proba()` with DP algorithm
   - Update state count calculations

2. **`methylation_hmm/emission_params.py`**
   - Add `compute_data_range()` for min/max
   - Store `data_min`, `data_max` in params

### Testing

- Re-run 4 configurations (binary/3way × single/pooled)
- Compare accuracy before/after
- Verify variable-length input handling

### Output

Results to `results/full_evaluation_profile_hmm/`
