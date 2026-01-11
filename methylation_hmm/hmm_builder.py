"""
Profile HMM Builder.

Constructs a DenseHMM with:
- Match and Insert states for each position
- Binary forks (C vs 5mC) at cytosine positions
- Transition matrix encoding the profile structure
"""

from typing import Dict, List, Tuple, Optional

import torch
import numpy as np
from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Normal

from .config import HMMConfig
from .kmer_model import KmerModel
from .distributions import DistributionFactory, StateInfo


class ProfileHMMBuilder:
    """Builds the complete Profile HMM."""

    def __init__(self, config: HMMConfig, kmer_model: KmerModel):
        """
        Initialize builder.

        Args:
            config: HMM configuration
            kmer_model: Loaded 9mer model
        """
        self.config = config
        self.kmer_model = kmer_model
        self.dist_factory = DistributionFactory(kmer_model, config)

        # These are set during build
        self.state_info: List[StateInfo] = []
        self.fork_indices: Dict[int, Dict[str, int]] = {}
        self.n_states: int = 0

    def build_model(self, sequence: str) -> DenseHMM:
        """
        Build complete Profile HMM for sequence.

        Args:
            sequence: Reference DNA sequence

        Returns:
            Configured DenseHMM ready for training/inference
        """
        # Step 1: Create emission distributions for all states
        distributions, self.state_info, self.fork_indices = \
            self.dist_factory.build_all_states(sequence)

        self.n_states = len(distributions)

        # Step 2: Build transition matrix
        edges = self._build_transition_matrix(sequence)

        # Step 3: Build start/end probabilities
        starts = self._build_start_probabilities()
        ends = self._build_end_probabilities(sequence)

        # Step 4: Create DenseHMM
        model = DenseHMM(
            distributions=distributions,
            edges=edges,
            starts=starts,
            ends=ends,
            max_iter=self.config.max_iterations,
            tol=self.config.convergence_threshold
        )

        print(f"Built Profile HMM with {self.n_states} states")
        return model

    def _build_transition_matrix(self, sequence: str) -> torch.Tensor:
        """
        Build dense transition matrix.

        Transition structure per position:
        - Match -> next Match (0.90 - self_loop)
        - Match -> Match self-loop (0.10)
        - Match -> next Insert (small prob)
        - Insert -> Insert self-loop (0.30)
        - Insert -> next Match (0.70)

        At fork positions:
        - Previous state -> C_match with p_canonical
        - Previous state -> 5mC_match with p_methylated

        Returns:
            Tensor of shape (n_states, n_states)
        """
        n = self.n_states
        edges = torch.zeros((n, n), dtype=torch.float32)

        cytosine_set = set(self.config.cytosine_positions)

        # Build state index mapping: position -> list of state indices
        pos_to_states = self._get_position_to_states()

        for pos in range(len(sequence)):
            current_states = pos_to_states[pos]
            next_pos = pos + 1

            if next_pos >= len(sequence):
                # Last position - transitions to end handled separately
                continue

            next_states = pos_to_states[next_pos]
            is_fork = pos in cytosine_set
            next_is_fork = next_pos in cytosine_set

            # Get state indices for current position
            if is_fork:
                match_c_idx = self.fork_indices[pos]['C']
                match_5mc_idx = self.fork_indices[pos]['5mC']
                insert_idx = match_5mc_idx + 1  # Insert follows 5mC
                match_indices = [match_c_idx, match_5mc_idx]
            else:
                # Non-fork: match is first, insert is second
                match_idx = current_states[0]
                insert_idx = current_states[1]
                match_indices = [match_idx]

            # Get next position's match state(s)
            if next_is_fork:
                next_match_c = self.fork_indices[next_pos]['C']
                next_match_5mc = self.fork_indices[next_pos]['5mC']
                next_insert = next_match_5mc + 1
            else:
                next_match = next_states[0]
                next_insert = next_states[1]

            # Add transitions from current match state(s)
            for m_idx in match_indices:
                # Self-loop for oversegmentation
                edges[m_idx, m_idx] = self.config.p_match_self_loop

                # Remaining probability goes forward
                forward_prob = 1.0 - self.config.p_match_self_loop

                if next_is_fork:
                    # Split to fork with equal priors
                    edges[m_idx, next_match_c] = forward_prob * self.config.p_canonical
                    edges[m_idx, next_match_5mc] = forward_prob * self.config.p_methylated
                else:
                    # Go to next match
                    edges[m_idx, next_match] = forward_prob * self.config.p_match
                    edges[m_idx, next_insert] = forward_prob * self.config.p_insert

            # Add transitions from current insert state
            edges[insert_idx, insert_idx] = self.config.p_insert_self_loop
            insert_forward = 1.0 - self.config.p_insert_self_loop

            if next_is_fork:
                edges[insert_idx, next_match_c] = insert_forward * self.config.p_canonical
                edges[insert_idx, next_match_5mc] = insert_forward * self.config.p_methylated
            else:
                edges[insert_idx, next_match] = insert_forward

        return edges

    def _build_start_probabilities(self) -> torch.Tensor:
        """
        Build start state probabilities.

        Only the first position's match state(s) can be started from.

        Returns:
            Tensor of shape (n_states,)
        """
        starts = torch.zeros(self.n_states, dtype=torch.float32)

        # First position is 0
        if 0 in self.config.cytosine_positions:
            # First position is a fork (unlikely but handle it)
            starts[self.fork_indices[0]['C']] = self.config.p_canonical
            starts[self.fork_indices[0]['5mC']] = self.config.p_methylated
        else:
            # Start at first match state (index 0)
            starts[0] = 1.0

        return starts

    def _build_end_probabilities(self, sequence: str) -> torch.Tensor:
        """
        Build end state probabilities.

        Only the last position's states can end.

        Returns:
            Tensor of shape (n_states,)
        """
        ends = torch.zeros(self.n_states, dtype=torch.float32)

        last_pos = len(sequence) - 1
        pos_to_states = self._get_position_to_states()
        last_states = pos_to_states[last_pos]

        # All last position states can end with equal probability
        for state_idx in last_states:
            ends[state_idx] = 1.0 / len(last_states)

        return ends

    def _get_position_to_states(self) -> Dict[int, List[int]]:
        """
        Map each position to its state indices.

        Returns:
            {position: [state_idx, ...]}
        """
        pos_to_states = {}
        current_idx = 0
        cytosine_set = set(self.config.cytosine_positions)

        for pos in range(self.config.sequence_length):
            if pos in cytosine_set:
                # Fork: 3 states (C_match, 5mC_match, insert)
                pos_to_states[pos] = [current_idx, current_idx + 1, current_idx + 2]
                current_idx += 3
            else:
                # Non-fork: 2 states (match, insert)
                pos_to_states[pos] = [current_idx, current_idx + 1]
                current_idx += 2

        return pos_to_states

    def get_fork_state_indices(self) -> Dict[int, Dict[str, int]]:
        """
        Get mapping of cytosine position to fork state indices.

        Returns:
            {position: {'C': state_idx, '5mC': state_idx}}
        """
        return self.fork_indices

    def get_state_info(self) -> List[StateInfo]:
        """Get metadata for all states."""
        return self.state_info

    def save_model_structure(self, filepath: str) -> None:
        """
        Save model structure to file for debugging.

        Args:
            filepath: Output path
        """
        with open(filepath, 'w') as f:
            f.write(f"# Profile HMM Structure\n")
            f.write(f"# States: {self.n_states}\n")
            f.write(f"# Forks: {len(self.fork_indices)}\n\n")

            f.write("## States\n")
            for idx, info in enumerate(self.state_info):
                f.write(f"{idx}\t{info.name}\t{info.state_type}\tpos={info.position}\n")

            f.write("\n## Fork Positions\n")
            for pos, indices in self.fork_indices.items():
                f.write(f"Position {pos}: C={indices['C']}, 5mC={indices['5mC']}\n")


def build_hmm_from_config(config: HMMConfig) -> Tuple[DenseHMM, ProfileHMMBuilder]:
    """
    Convenience function to build HMM from config.

    Args:
        config: HMM configuration with paths set

    Returns:
        (model, builder) tuple
    """
    from .data_loader import load_reference_sequence

    # Load reference sequence
    sequence = load_reference_sequence(config.reference_fasta_path)
    config.reference_sequence = sequence
    config.sequence_length = len(sequence)

    # Load kmer model
    kmer_model = KmerModel(config.kmer_model_path)

    # Build HMM
    builder = ProfileHMMBuilder(config, kmer_model)
    model = builder.build_model(sequence)

    return model, builder
