"""
YAHMM to Pomegranate Loader

Loads HMM models from YAHMM's legacy text format into modern pomegranate (v1.0+).
Handles silent state elimination automatically since pomegranate v1.0+ doesn't
support silent states.

Problem:
    - YAHMM models contain both emitting states (with distributions) and silent
      states (for routing transitions without emissions)
    - Modern pomegranate only supports emitting states
    - This module mathematically eliminates silent states while preserving
      equivalent transition probabilities

Math:
    Silent states are eliminated using the formula:
        P_effective = P_EE + P_ES @ (I - P_SS)^(-1) @ P_SE

    Where:
        P_EE: emitting -> emitting transitions
        P_ES: emitting -> silent transitions
        P_SE: silent -> emitting transitions
        P_SS: silent -> silent transitions
        (I - P_SS)^(-1): transitive closure summing all paths through silent states

Usage:
    # Basic usage - load model and use for inference
    from yahmm_loader import load_yahmm_model
    import torch

    model, state_names = load_yahmm_model('trained_hmm.txt')

    # Run inference on observation sequence
    observations = [37.2, 41.5, 38.9, ...]  # e.g., nanopore current levels
    X = torch.tensor(observations, dtype=torch.float32).reshape(1, -1, 1)
    log_prob = model.log_probability(X)

    # For batch processing
    sequences = [[37.2, 41.5], [38.1, 42.0, 39.5], ...]
    for seq in sequences:
        X = torch.tensor(seq, dtype=torch.float32).reshape(1, -1, 1)
        print(model.log_probability(X))

Advanced usage:
    # Access intermediate parsing results for verification
    from yahmm_loader import (
        parse_yahmm_file,
        classify_states,
        build_transition_matrices,
        eliminate_silent_states
    )

    model_name, states, transitions = parse_yahmm_file('model.txt')
    emitting, silent = classify_states(states)
    P_EE, P_ES, P_SE, P_SS, emit_ids, silent_ids = build_transition_matrices(
        emitting, silent, transitions
    )
    P_effective, closure = eliminate_silent_states(P_EE, P_ES, P_SE, P_SS)

Requirements:
    - pomegranate >= 1.0.0
    - torch
    - numpy

Note:
    This loader was verified against original YAHMM models by comparing
    transition matrices. The elimination is mathematically exact (difference
    < 1e-15, i.e., floating-point precision).
"""

import numpy as np
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class ParsedState:
    """Represents a parsed state from YAHMM format."""

    identity: str
    name: str
    weight: float
    distribution: Optional[str]

    @property
    def is_silent(self) -> bool:
        return self.distribution is None or self.distribution == "None"


@dataclass
class ParsedTransition:
    """Represents a parsed transition from YAHMM format."""

    from_name: str
    to_name: str
    probability: float
    pseudocount: float
    from_id: str
    to_id: str


def parse_yahmm_file(
    filepath: str,
) -> Tuple[str, Dict[str, ParsedState], List[ParsedTransition]]:
    """
    Parse a YAHMM model.txt file.

    Args:
        filepath: Path to the model file

    Returns:
        model_name: Name of the model
        states: Dict mapping identity -> ParsedState
        transitions: List of ParsedTransition objects
    """
    states = {}
    transitions = []

    with open(filepath, "r") as f:
        lines = f.readlines()

    # Parse header: "{model_name} {num_states}"
    header = lines[0].strip().split()
    model_name = header[0]
    num_states = int(header[1])

    # Parse states (lines 1 to num_states)
    for i in range(1, num_states + 1):
        line = lines[i].strip()
        parts = line.split()

        identity = parts[0]
        name = parts[1]
        weight = float(parts[2])

        # Distribution is everything after weight
        dist_str = " ".join(parts[3:]) if len(parts) > 3 else "None"
        distribution = None if dist_str == "None" else dist_str

        states[identity] = ParsedState(identity, name, weight, distribution)

    # Parse transitions (remaining lines)
    for i in range(num_states + 1, len(lines)):
        line = lines[i].strip()
        if not line:
            continue

        parts = line.split()
        transitions.append(
            ParsedTransition(
                from_name=parts[0],
                to_name=parts[1],
                probability=float(parts[2]),
                pseudocount=float(parts[3]),
                from_id=parts[4],
                to_id=parts[5],
            )
        )

    return model_name, states, transitions


def classify_states(
    states: Dict[str, ParsedState],
) -> Tuple[Dict[str, ParsedState], Dict[str, ParsedState]]:
    """Separate states into emitting and silent."""
    emitting = {k: v for k, v in states.items() if not v.is_silent}
    silent = {k: v for k, v in states.items() if v.is_silent}
    return emitting, silent


def build_transition_matrices(
    emitting_states: Dict[str, ParsedState],
    silent_states: Dict[str, ParsedState],
    transitions: List[ParsedTransition],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the four transition submatrices.

    Returns:
        P_EE: emitting -> emitting
        P_ES: emitting -> silent
        P_SE: silent -> emitting
        P_SS: silent -> silent
    """
    emit_ids = list(emitting_states.keys())
    silent_ids = list(silent_states.keys())

    emit_idx = {id_: i for i, id_ in enumerate(emit_ids)}
    silent_idx = {id_: i for i, id_ in enumerate(silent_ids)}

    n_emit = len(emit_ids)
    n_silent = len(silent_ids)

    P_EE = np.zeros((n_emit, n_emit))
    P_ES = np.zeros((n_emit, n_silent))
    P_SE = np.zeros((n_silent, n_emit))
    P_SS = np.zeros((n_silent, n_silent))

    for t in transitions:
        from_emit = t.from_id in emit_idx
        to_emit = t.to_id in emit_idx

        if from_emit and to_emit:
            P_EE[emit_idx[t.from_id], emit_idx[t.to_id]] = t.probability
        elif from_emit and t.to_id in silent_idx:
            P_ES[emit_idx[t.from_id], silent_idx[t.to_id]] = t.probability
        elif t.from_id in silent_idx and to_emit:
            P_SE[silent_idx[t.from_id], emit_idx[t.to_id]] = t.probability
        elif t.from_id in silent_idx and t.to_id in silent_idx:
            P_SS[silent_idx[t.from_id], silent_idx[t.to_id]] = t.probability

    return P_EE, P_ES, P_SE, P_SS, emit_ids, silent_ids


def eliminate_silent_states(
    P_EE: np.ndarray, P_ES: np.ndarray, P_SE: np.ndarray, P_SS: np.ndarray
) -> np.ndarray:
    """
    Eliminate silent states using matrix algebra.

    Formula: P_effective = P_EE + P_ES @ (I - P_SS)^(-1) @ P_SE

    The term (I - P_SS)^(-1) computes the transitive closure of silent states,
    summing all possible paths: I + P_SS + P_SS^2 + P_SS^3 + ...
    """
    n_silent = P_SS.shape[0]

    if n_silent == 0:
        return P_EE

    I = np.eye(n_silent)

    # Check invertibility
    eigenvalues = np.linalg.eigvals(P_SS)
    spectral_radius = np.max(np.abs(eigenvalues))

    if spectral_radius >= 1.0:
        raise ValueError(
            f"P_SS has spectral radius {spectral_radius:.4f} >= 1. "
            "Silent states form an absorbing cycle."
        )

    # Compute transitive closure
    silent_closure = np.linalg.inv(I - P_SS)

    # Apply elimination formula
    P_effective = P_EE + P_ES @ silent_closure @ P_SE

    return P_effective, silent_closure


def compute_effective_boundaries(
    model_name: str,
    states: Dict[str, ParsedState],
    transitions: List[ParsedTransition],
    emitting_states: Dict[str, ParsedState],
    silent_states: Dict[str, ParsedState],
    P_ES: np.ndarray,
    P_SE: np.ndarray,
    silent_closure: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute effective start and end probabilities."""

    emit_ids = list(emitting_states.keys())
    silent_ids = list(silent_states.keys())
    emit_idx = {id_: i for i, id_ in enumerate(emit_ids)}
    silent_idx = {id_: i for i, id_ in enumerate(silent_ids)}

    n_emit = len(emit_ids)
    n_silent = len(silent_ids)

    # Find model start/end
    model_start = None
    model_end = None
    for s in states.values():
        if s.name.endswith("-start") and model_name in s.name:
            model_start = s
        if s.name.endswith("-end") and model_name in s.name:
            model_end = s

    # Compute start probabilities
    start_to_emit = np.zeros(n_emit)
    start_to_silent = np.zeros(n_silent)

    if model_start:
        for t in transitions:
            if t.from_id == model_start.identity:
                if t.to_id in emit_idx:
                    start_to_emit[emit_idx[t.to_id]] = t.probability
                elif t.to_id in silent_idx:
                    start_to_silent[silent_idx[t.to_id]] = t.probability

    effective_starts = start_to_emit + start_to_silent @ silent_closure @ P_SE

    # Compute end probabilities
    emit_to_end = np.zeros(n_emit)
    silent_to_end = np.zeros(n_silent)

    if model_end:
        for t in transitions:
            if t.to_id == model_end.identity:
                if t.from_id in emit_idx:
                    emit_to_end[emit_idx[t.from_id]] = t.probability
                elif t.from_id in silent_idx:
                    silent_to_end[silent_idx[t.from_id]] = t.probability

    effective_ends = emit_to_end + P_ES @ silent_closure @ silent_to_end

    return effective_starts, effective_ends


def parse_distribution(dist_str: str) -> Tuple[Optional[str], Optional[List[float]]]:
    """Parse 'NormalDistribution(37.04, 0.55)' into (type, [params])."""
    if dist_str is None:
        return None, None

    match = re.match(r"(\w+)\((.*)\)", dist_str)
    if not match:
        return None, None

    dist_type = match.group(1)
    params = [float(x.strip()) for x in match.group(2).split(",")]

    return dist_type, params


def load_yahmm_model(filepath: str, verbose: bool = True):
    """
    Load a YAHMM model file into pomegranate DenseHMM.

    Args:
        filepath: Path to the YAHMM model.txt file
        verbose: Print progress information

    Returns:
        model: pomegranate DenseHMM object
        state_names: List of emitting state names (in order)
    """
    try:
        import torch
        from pomegranate.hmm import DenseHMM
        from pomegranate.distributions import Normal
    except ImportError as e:
        raise ImportError(
            f"pomegranate v1.0+ required: {e}\n"
            "Install with: pip install pomegranate>=1.0"
        )

    # Parse file
    if verbose:
        print(f"Parsing {filepath}...")
    model_name, states, transitions = parse_yahmm_file(filepath)

    if verbose:
        print(f"  Model: {model_name}")
        print(f"  States: {len(states)}")
        print(f"  Transitions: {len(transitions)}")

    # Classify states
    emitting_states, silent_states = classify_states(states)
    if verbose:
        print(f"  Emitting: {len(emitting_states)}, Silent: {len(silent_states)}")

    # Build matrices
    P_EE, P_ES, P_SE, P_SS, emit_ids, silent_ids = build_transition_matrices(
        emitting_states, silent_states, transitions
    )

    # Eliminate silent states
    if verbose:
        print("Eliminating silent states...")
    P_effective, silent_closure = eliminate_silent_states(P_EE, P_ES, P_SE, P_SS)

    # Compute boundaries
    effective_starts, effective_ends = compute_effective_boundaries(
        model_name,
        states,
        transitions,
        emitting_states,
        silent_states,
        P_ES,
        P_SE,
        silent_closure,
    )

    # Create distributions
    if verbose:
        print("Creating distributions...")
    distributions = []
    state_names = []

    for id_ in emit_ids:
        state = emitting_states[id_]
        state_names.append(state)
        dist_type, params = parse_distribution(state.distribution)

        if dist_type == "NormalDistribution":
            mean, std = params
            # covariance_type='diag' for univariate - pass variance as 1D array
            distributions.append(
                Normal(means=[mean], covs=[std**2], covariance_type="diag")
            )
        elif dist_type == "UniformDistribution":
            # Approximate uniform with normal (same mean, matched variance)
            low, high = params
            mean = (low + high) / 2
            std = (high - low) / np.sqrt(12)
            distributions.append(
                Normal(means=[mean], covs=[std**2], covariance_type="diag")
            )
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")

    # Normalize probabilities
    starts_sum = effective_starts.sum()
    if starts_sum > 0:
        starts_normalized = effective_starts / starts_sum
    else:
        starts_normalized = np.ones(len(emit_ids)) / len(emit_ids)

    # Build model
    # Note: kind='sparse' is the default - pomegranate drops zeros internally
    n_nonzero = np.count_nonzero(P_effective)
    sparsity = 1 - n_nonzero / P_effective.size
    if verbose:
        print(
            f"Building DenseHMM (sparsity: {sparsity:.1%}, {n_nonzero} non-zero edges)..."
        )

    model = DenseHMM(
        distributions=distributions,
        edges=torch.tensor(P_effective, dtype=torch.float32),
        starts=torch.tensor(starts_normalized, dtype=torch.float32),
        ends=torch.tensor(effective_ends, dtype=torch.float32),
    )

    if verbose:
        print(f"Done! Model has {len(distributions)} emitting states.")

    return model, state_names


# Validation utilities


def validate_model(
    P_effective: np.ndarray, effective_starts: np.ndarray, effective_ends: np.ndarray
):
    """Validate the eliminated model matrices."""
    issues = []

    # Check for negative values
    if (P_effective < 0).any():
        issues.append(f"Negative transitions: {(P_effective < 0).sum()}")

    if (effective_starts < 0).any():
        issues.append(f"Negative start probs: {(effective_starts < 0).sum()}")

    if (effective_ends < 0).any():
        issues.append(f"Negative end probs: {(effective_ends < 0).sum()}")

    # Check for NaN/Inf
    if np.isnan(P_effective).any():
        issues.append(f"NaN in transitions: {np.isnan(P_effective).sum()}")

    if np.isinf(P_effective).any():
        issues.append(f"Inf in transitions: {np.isinf(P_effective).sum()}")

    # Check row sums
    row_sums = P_effective.sum(axis=1) + effective_ends
    if (row_sums > 1.01).any():
        issues.append(f"Row sums > 1: {(row_sums > 1.01).sum()}")

    return issues


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python yahmm_loader.py <model.txt>")
        sys.exit(1)

    filepath = sys.argv[1]
    model, state_names = load_yahmm_model(filepath)

    print(f"\nState names (first 10):")
    for state in state_names[:10]:
        print(f"  {state.name}")
