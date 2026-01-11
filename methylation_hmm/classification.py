"""
Methylation classification using forward-backward algorithm.

Computes posterior probabilities for C vs 5mC at each fork position
based on expected transition counts through the fork states.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import torch
import numpy as np
from pomegranate.hmm import DenseHMM

from .config import HMMConfig
from .hmm_builder import ProfileHMMBuilder
from .data_loader import SegmentedRead


@dataclass
class ClassificationResult:
    """Classification result for a single read at one cytosine position."""
    read_id: str
    position: int           # Cytosine position in reference
    p_canonical: float      # P(C)
    p_methylated: float     # P(5mC)
    call: str              # 'C' or '5mC'
    confidence: float       # abs(p_methylated - 0.5) * 2, range [0,1]
    log_probability: float  # Log probability of sequence


@dataclass
class ReadClassification:
    """Full classification for a read across all cytosine positions."""
    read_id: str
    positions: List[ClassificationResult]
    filter_score: float     # Product of max(p_C, p_5mC) across forks
    log_probability: float  # Total log probability


class MethylationClassifier:
    """Classify cytosines using forward-backward algorithm."""

    def __init__(
        self,
        model: DenseHMM,
        builder: ProfileHMMBuilder,
        config: HMMConfig
    ):
        """
        Initialize classifier.

        Args:
            model: Trained or initialized DenseHMM
            builder: ProfileHMMBuilder with state metadata
            config: HMM configuration
        """
        self.model = model
        self.builder = builder
        self.config = config
        self.fork_indices = builder.get_fork_state_indices()
        self.state_info = builder.get_state_info()

    def classify_read(
        self,
        segments: np.ndarray,
        read_id: str = "unknown"
    ) -> ReadClassification:
        """
        Classify all cytosine positions in a read.

        Uses forward-backward to get posterior probabilities
        for C vs 5mC at each fork position.

        Args:
            segments: Array of z-scored current values
            read_id: Identifier for the read

        Returns:
            ReadClassification with results for all positions
        """
        # Convert to tensor format expected by pomegranate
        # Shape: (1, seq_len, 1) for single sequence with 1 feature
        X = torch.tensor(segments, dtype=torch.float32).reshape(1, -1, 1)

        # Run forward-backward
        logp, responsibilities = self._forward_backward(X)

        # Compute posteriors at each fork
        results = []
        confidences = []

        for pos in self.config.cytosine_positions:
            if pos not in self.fork_indices:
                continue

            p_c, p_5mc = self._compute_fork_posteriors(responsibilities, pos)

            # Make call
            call = '5mC' if p_5mc > p_c else 'C'
            confidence = abs(p_5mc - 0.5) * 2  # Scale to [0, 1]
            confidences.append(max(p_c, p_5mc))

            results.append(ClassificationResult(
                read_id=read_id,
                position=pos,
                p_canonical=p_c,
                p_methylated=p_5mc,
                call=call,
                confidence=confidence,
                log_probability=logp.item()
            ))

        # Filter score = product of max posteriors (measures path confidence)
        filter_score = np.prod(confidences) if confidences else 0.0

        return ReadClassification(
            read_id=read_id,
            positions=results,
            filter_score=filter_score,
            log_probability=logp.item()
        )

    def _forward_backward(
        self,
        X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run forward-backward algorithm.

        Args:
            X: Input tensor of shape (batch, seq_len, features)

        Returns:
            (log_probability, responsibilities)
            - log_probability: scalar log P(X | model)
            - responsibilities: (seq_len, n_states) posterior state probs
        """
        # Use pomegranate's forward-backward
        # This returns log probability and responsibilities (gamma)
        self.model.eval()

        with torch.no_grad():
            # Forward pass to get log probability
            logp = self.model.log_probability(X)

            # Get responsibilities (posterior state probabilities)
            # gamma[t, j] = P(state_j at time t | X)
            responsibilities = self.model.predict_proba(X)

        return logp, responsibilities[0]  # Remove batch dimension

    def _compute_fork_posteriors(
        self,
        responsibilities: torch.Tensor,
        position: int
    ) -> Tuple[float, float]:
        """
        Compute posterior probability for C vs 5mC at a fork.

        Strategy: Sum responsibilities for C and 5mC states,
        then normalize.

        Args:
            responsibilities: (seq_len, n_states) posterior probs
            position: Cytosine position

        Returns:
            (p_canonical, p_methylated)
        """
        c_idx = self.fork_indices[position]['C']
        mc_idx = self.fork_indices[position]['5mC']

        # Sum responsibilities across all time steps
        # This gives total "flow" through each state
        c_flow = responsibilities[:, c_idx].sum().item()
        mc_flow = responsibilities[:, mc_idx].sum().item()

        # Normalize to get posteriors
        total = c_flow + mc_flow
        if total < 1e-10:
            # No flow through fork (shouldn't happen with good data)
            return 0.5, 0.5

        p_c = c_flow / total
        p_5mc = mc_flow / total

        return p_c, p_5mc

    def classify_batch(
        self,
        reads: List[SegmentedRead],
        filter_threshold: Optional[float] = None
    ) -> List[ReadClassification]:
        """
        Classify multiple reads.

        Args:
            reads: List of SegmentedRead objects
            filter_threshold: Optional filter score threshold

        Returns:
            List of ReadClassification (filtered if threshold set)
        """
        if filter_threshold is None:
            filter_threshold = self.config.filter_score_threshold

        results = []
        filtered_count = 0

        for read in reads:
            result = self.classify_read(read.segments, read.read_id)

            if result.filter_score >= filter_threshold:
                results.append(result)
            else:
                filtered_count += 1

        print(f"Classified {len(results)} reads "
              f"(filtered {filtered_count} with score < {filter_threshold})")

        return results

    def results_to_dataframe(
        self,
        classifications: List[ReadClassification]
    ):
        """
        Convert classification results to DataFrame.

        Returns:
            DataFrame with columns:
            read_id, position, p_canonical, p_methylated, call, confidence, filter_score
        """
        import pandas as pd

        rows = []
        for result in classifications:
            for pos_result in result.positions:
                rows.append({
                    'read_id': pos_result.read_id,
                    'position': pos_result.position,
                    'p_canonical': pos_result.p_canonical,
                    'p_methylated': pos_result.p_methylated,
                    'call': pos_result.call,
                    'confidence': pos_result.confidence,
                    'filter_score': result.filter_score,
                    'log_probability': pos_result.log_probability
                })

        return pd.DataFrame(rows)


def classify_from_tsv(
    model: DenseHMM,
    builder: ProfileHMMBuilder,
    config: HMMConfig,
    tsv_path: str,
    output_path: Optional[str] = None
):
    """
    Convenience function to classify reads from TSV file.

    Args:
        model: Trained HMM
        builder: HMM builder with metadata
        config: Configuration
        tsv_path: Path to input TSV
        output_path: Optional path to save results CSV

    Returns:
        DataFrame with classification results
    """
    from .data_loader import DataLoader

    # Load and preprocess data
    loader = DataLoader(
        min_segments=config.min_segments_per_read,
        max_segments=config.max_segments_per_read
    )
    reads = loader.load_and_preprocess(tsv_path, normalize=True)

    # Classify
    classifier = MethylationClassifier(model, builder, config)
    results = classifier.classify_batch(reads)

    # Convert to DataFrame
    df = classifier.results_to_dataframe(results)

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} classifications to {output_path}")

    return df
