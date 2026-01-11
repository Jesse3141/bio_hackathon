"""
Emission distribution factory for Profile HMM states.

Creates Normal distributions for match states based on the 9mer model,
with modifications for 5mC detection at fork positions.
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass

import torch
from pomegranate.distributions import Normal

from .config import HMMConfig
from .kmer_model import KmerModel


@dataclass
class StateInfo:
    """Metadata about an HMM state."""
    position: int
    state_type: str       # 'match', 'match_C', 'match_5mC', 'insert'
    distribution: Normal
    name: str


class DistributionFactory:
    """Creates emission distributions for HMM states."""

    def __init__(self, kmer_model: KmerModel, config: HMMConfig):
        """
        Initialize factory.

        Args:
            kmer_model: Loaded 9mer model
            config: HMM configuration
        """
        self.kmer_model = kmer_model
        self.config = config

    def create_match_distribution(
        self,
        sequence: str,
        position: int,
        modification: str = 'C'
    ) -> Normal:
        """
        Create Normal distribution for a match state.

        Args:
            sequence: Reference sequence
            position: 0-indexed position
            modification: 'C' or '5mC'

        Returns:
            pomegranate Normal distribution
        """
        mean, std = self.kmer_model.get_emission_params(
            sequence=sequence,
            position=position,
            modification=modification,
            methylation_shift=self.config.methylation_shift,
            methylation_std_factor=self.config.methylation_std_factor
        )

        # Pomegranate expects tensors
        return Normal(
            means=torch.tensor([mean]),
            covs=torch.tensor([[std ** 2]])  # Variance, not std
        )

    def create_insert_distribution(self) -> Normal:
        """
        Create wide Normal distribution for insert states.

        Insert states handle extra segments (noise, blips).
        Use a wide distribution centered at 0.

        Returns:
            pomegranate Normal distribution
        """
        return Normal(
            means=torch.tensor([0.0]),
            covs=torch.tensor([[self.config.insert_std ** 2]])
        )

    def build_all_states(
        self,
        sequence: str
    ) -> Tuple[List[Normal], List[StateInfo], Dict[int, Dict[str, int]]]:
        """
        Build all emission distributions for the sequence.

        Creates:
        - Match + Insert states for non-fork positions
        - Match_C + Match_5mC + Insert states for fork positions

        Args:
            sequence: Reference DNA sequence

        Returns:
            (distributions, state_info, fork_indices)
            - distributions: List of Normal distributions
            - state_info: List of StateInfo metadata
            - fork_indices: {position: {'C': idx, '5mC': idx}}
        """
        distributions = []
        state_info = []
        fork_indices = {}

        state_idx = 0
        cytosine_set = set(self.config.cytosine_positions)

        for pos in range(len(sequence)):
            if pos in cytosine_set:
                # Fork position: create C and 5mC match states
                fork_indices[pos] = {}

                # C match state
                dist_c = self.create_match_distribution(sequence, pos, 'C')
                distributions.append(dist_c)
                state_info.append(StateInfo(
                    position=pos,
                    state_type='match_C',
                    distribution=dist_c,
                    name=f'M_C:{pos}'
                ))
                fork_indices[pos]['C'] = state_idx
                state_idx += 1

                # 5mC match state
                dist_5mc = self.create_match_distribution(sequence, pos, '5mC')
                distributions.append(dist_5mc)
                state_info.append(StateInfo(
                    position=pos,
                    state_type='match_5mC',
                    distribution=dist_5mc,
                    name=f'M_5mC:{pos}'
                ))
                fork_indices[pos]['5mC'] = state_idx
                state_idx += 1

                # Insert state for this position
                dist_ins = self.create_insert_distribution()
                distributions.append(dist_ins)
                state_info.append(StateInfo(
                    position=pos,
                    state_type='insert',
                    distribution=dist_ins,
                    name=f'I:{pos}'
                ))
                state_idx += 1

            else:
                # Non-fork position: single match + insert
                dist_match = self.create_match_distribution(sequence, pos, 'C')
                distributions.append(dist_match)
                state_info.append(StateInfo(
                    position=pos,
                    state_type='match',
                    distribution=dist_match,
                    name=f'M:{pos}'
                ))
                state_idx += 1

                dist_ins = self.create_insert_distribution()
                distributions.append(dist_ins)
                state_info.append(StateInfo(
                    position=pos,
                    state_type='insert',
                    distribution=dist_ins,
                    name=f'I:{pos}'
                ))
                state_idx += 1

        print(f"Created {len(distributions)} states for {len(sequence)}bp sequence")
        print(f"  - {len(fork_indices)} fork positions with binary C/5mC branches")

        return distributions, state_info, fork_indices


def get_state_indices_by_type(
    state_info: List[StateInfo]
) -> Dict[str, List[int]]:
    """
    Group state indices by type.

    Returns:
        {'match': [...], 'match_C': [...], 'match_5mC': [...], 'insert': [...]}
    """
    indices = {
        'match': [],
        'match_C': [],
        'match_5mC': [],
        'insert': []
    }

    for idx, info in enumerate(state_info):
        indices[info.state_type].append(idx)

    return indices
