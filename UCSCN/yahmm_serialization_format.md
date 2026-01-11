# YAHMM Model Serialization Format

This document explains how `yahmm.Model.read()` deserializes a saved HMM model, to inform porting to pomegranate.

## Quick Reference

```python
from yahmm import Model

with open('untrained_hmm.txt', 'r') as f:
    model = Model.read(f)
```

## File Format Overview

The serialization format is **line-based plaintext** with three sections:

```
┌─────────────────────────────────────────────────────────────────┐
│  Line 1:        HEADER                                          │
├─────────────────────────────────────────────────────────────────┤
│  Lines 2 to N+1: STATES (N = state count from header)           │
├─────────────────────────────────────────────────────────────────┤
│  Lines N+2 to EOF: TRANSITIONS                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Section 1: Header Line

**Format:** `{model_name} {num_states}`

**Example:**
```
EpigeneticsHMM-54 1070
```

**Meaning:**
- `EpigeneticsHMM-54`: Model name (used to identify start/end states)
- `1070`: Total number of states in the model

## Section 2: State Lines

**Format:** `{identity} {name} {weight} {distribution}`

**Examples:**
```
363863352 D-:1 1.0 None
363863568 M-:1 1.0 NormalDistribution(37.0422418529, 0.546369746415)
363863712 I-:1 1.0 UniformDistribution(0, 90)
363862056 EpigeneticsHMM-54-end 1.0 None
363861984 EpigeneticsHMM-54-start 1.0 None
```

**Field breakdown:**

| Field | Type | Description |
|-------|------|-------------|
| `identity` | int (memory address) | Unique identifier for state (Python object id at write time) |
| `name` | string | State name (spaces replaced with underscores) |
| `weight` | float | State weight (usually 1.0) |
| `distribution` | string or `None` | Distribution repr string, or `None` for silent states |

**State types by naming convention:**

| Prefix | Type | Distribution |
|--------|------|--------------|
| `D-*` | Delete (silent) | `None` |
| `M-*` | Match (emitting) | `NormalDistribution(mean, std)` |
| `MO-*` | Overseg match | `NormalDistribution(mean, std)` |
| `I-*` | Insert | `UniformDistribution(low, high)` |
| `U-*` | Underseg blend | `NormalDistribution(blend_mean, blend_std)` |
| `*-start` | Model start (silent) | `None` |
| `*-end` | Model end (silent) | `None` |
| `b*s*` / `b*e*` | Board ports (silent) | `None` |

## Section 3: Transition Lines

**Format:** `{from_name} {to_name} {probability} {pseudocount} {from_id} {to_id}`

**Examples:**
```
M-X:39 M-X:39-end 0.9 0.9 383780040 383779968
M-X:39 M-X:39 0.1 0.1 383780040 383780040
I-X:39 D-X:40 0.05 1.0 383780184 383805192
b:36e2 M-CAT:37-start 0.333333333333 1000000.0 374050392 383739224
```

**Field breakdown:**

| Field | Type | Description |
|-------|------|-------------|
| `from_name` | string | Source state name |
| `to_name` | string | Destination state name |
| `probability` | float | Transition probability (linear, not log) |
| `pseudocount` | float | Pseudocount for training (often same as prob, or large like `1e6`) |
| `from_id` | int | Identity of source state |
| `to_id` | int | Identity of destination state |

## How `Model.read()` Works

```python
@classmethod
def read(cls, stream):
    # 1. Parse header
    line = stream.readline()
    name, n_states = line.split()[0], int(line.split()[1])

    # 2. Read all states into dict keyed by identity
    states = {}
    for _ in range(n_states):
        state = State.read(stream)  # Parses one state line
        states[state.identity] = state

    # 3. Find start/end states by naming convention
    start_state = [s for s in states.values() if s.name == name + "-start"][0]
    end_state = [s for s in states.values() if s.name == name + "-end"][0]

    # 4. Create model with identified start/end
    model = Model(name=name, start=start_state, end=end_state)

    # 5. Add all other states
    for state in states.values():
        if state not in (start_state, end_state):
            model.add_state(state)

    # 6. Parse transitions
    for line in stream:
        parts = line.strip().split()
        from_name, to_name = parts[0], parts[1]
        probability = float(parts[2])
        pseudocount = float(parts[3])
        from_id, to_id = parts[4], parts[5]

        # Lookup states by identity
        from_state = states[from_id]
        to_state = states[to_id]

        model.add_transition(from_state, to_state, probability, pseudocount)

    # 7. Finalize
    model.bake(merge=None)
    return model
```

## How `State.read()` Works

```python
@classmethod
def read(cls, stream):
    line = stream.readline()
    parts = line.strip().split()

    identity = parts[0]
    name = parts[1]  # underscores -> spaces in original
    weight = parts[2]
    distribution = ' '.join(parts[3:])  # May contain spaces

    # CRITICAL: Uses eval() to reconstruct distribution!
    return eval("State({}, name='{}', weight={}, identity='{}')".format(
        distribution, name, weight, identity
    ))
```

**Key insight:** Distribution strings like `NormalDistribution(37.04, 0.55)` are directly evaluated as Python code to reconstruct the object. This requires all distribution classes to be in scope.

## Distribution Classes Used

```python
# Silent states
None

# Match/emitting states
NormalDistribution(mean, std)

# Insert states
UniformDistribution(low, high)

# (Also supports GaussianKernelDensity for trained models)
GaussianKernelDensity(points_list, bandwidth)
```

## Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                        untrained_hmm.txt                            │
│                                                                     │
│  "EpigeneticsHMM-54 1070"                                          │
│  "363863352 D-:1 1.0 None"              ─┐                         │
│  "363863568 M-:1 1.0 NormalDist(...)"    │ State lines             │
│  ...                                     │ (1070 total)            │
│  "363862056 EpigeneticsHMM-54-end ..."  ─┘                         │
│  "M-X:39 M-X:39-end 0.9 0.9 ..."        ─┐                         │
│  ...                                     │ Transition lines        │
│  "bhmC:33e7 M-hmC:33-start 0.9 ..."     ─┘ (~2800 total)           │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Model.read()                                 │
│                                                                     │
│  1. Parse header → name="EpigeneticsHMM-54", n=1070                │
│  2. For each state line:                                            │
│     - State.read() → eval() distribution string                     │
│     - Store in dict by identity                                     │
│  3. Find start/end by name matching                                 │
│  4. Create Model(name, start, end)                                  │
│  5. Add remaining states                                            │
│  6. Parse transitions → add_transition(from, to, prob, pseudo)      │
│  7. model.bake(merge=None)                                          │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     yahmm.Model object                              │
│                                                                     │
│  .name = "EpigeneticsHMM-54"                                       │
│  .states = [State, State, ...] (1070 states)                       │
│  .start = State(name="EpigeneticsHMM-54-start")                    │
│  .end = State(name="EpigeneticsHMM-54-end")                        │
│  .graph = networkx.DiGraph (nodes=states, edges=transitions)       │
│                                                                     │
│  Methods: forward_backward(), viterbi(), train(), ...              │
└─────────────────────────────────────────────────────────────────────┘
```

## For Pomegranate Port

Key mappings to investigate:

| yahmm | pomegranate equivalent |
|-------|----------------------|
| `Model` | `HiddenMarkovModel` |
| `State(distribution, name)` | `State(distribution, name)` |
| `NormalDistribution(mean, std)` | `NormalDistribution(mean, std)` |
| `UniformDistribution(low, high)` | `UniformDistribution(low, high)` |
| `model.add_state(state)` | `model.add_state(state)` |
| `model.add_transition(a, b, prob)` | `model.add_transition(a, b, prob)` |
| `model.bake()` | `model.bake()` |
| `model.forward_backward(seq)` | `model.forward_backward(seq)` |

The serialization format is custom to yahmm, so you'll need to write a custom loader for pomegranate that:
1. Parses the same text format
2. Maps distribution class names
3. Creates pomegranate State/Model objects
4. Handles the pseudocount parameter (may need different approach)
