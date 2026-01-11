"""
Baum-Welch training for the Profile HMM.

Trains emission parameters using labeled control and methylated data,
with filter score thresholding to use only high-confidence reads.
"""

from typing import List, Tuple, Optional
from pathlib import Path

import torch
import numpy as np
from pomegranate.hmm import DenseHMM

from .config import HMMConfig
from .hmm_builder import ProfileHMMBuilder
from .data_loader import DataLoader, SegmentedRead
from .classification import MethylationClassifier


class BaumWelchTrainer:
    """Baum-Welch training for the Profile HMM."""

    def __init__(
        self,
        model: DenseHMM,
        builder: ProfileHMMBuilder,
        config: HMMConfig
    ):
        """
        Initialize trainer.

        Args:
            model: Initialized DenseHMM
            builder: HMM builder with metadata
            config: Training configuration
        """
        self.model = model
        self.builder = builder
        self.config = config
        self.classifier = MethylationClassifier(model, builder, config)
        self.training_history: List[float] = []

    def prepare_training_data(
        self,
        reads: List[SegmentedRead]
    ) -> torch.Tensor:
        """
        Convert reads to tensor format for training.

        Args:
            reads: List of SegmentedRead objects

        Returns:
            Tensor of shape (n_reads, max_len, 1)
        """
        # Find max length
        max_len = max(len(r) for r in reads)

        # Pad sequences
        n_reads = len(reads)
        X = torch.zeros((n_reads, max_len, 1), dtype=torch.float32)

        for i, read in enumerate(reads):
            length = len(read)
            X[i, :length, 0] = torch.tensor(read.segments, dtype=torch.float32)

        return X

    def filter_by_score(
        self,
        reads: List[SegmentedRead],
        threshold: Optional[float] = None
    ) -> List[SegmentedRead]:
        """
        Filter reads by filter score.

        Only keeps reads with high confidence paths through all forks.

        Args:
            reads: List of reads
            threshold: Filter score threshold (default from config)

        Returns:
            Filtered list of reads
        """
        if threshold is None:
            threshold = self.config.filter_score_threshold

        filtered = []

        for read in reads:
            result = self.classifier.classify_read(read.segments, read.read_id)
            if result.filter_score >= threshold:
                filtered.append(read)

        print(f"Filter score filtering: {len(filtered)}/{len(reads)} reads "
              f"(threshold={threshold})")

        return filtered

    def train(
        self,
        control_reads: List[SegmentedRead],
        methylated_reads: Optional[List[SegmentedRead]] = None,
        n_iterations: Optional[int] = None,
        filter_before_training: bool = True
    ) -> DenseHMM:
        """
        Train model using labeled data.

        Strategy:
        1. Optionally filter by score to use only confident reads
        2. Combine control and methylated data
        3. Run Baum-Welch for n_iterations

        Note: The fork structure means control reads will naturally
        have higher posterior through C states, and methylated reads
        through 5mC states, which guides the emission updates.

        Args:
            control_reads: Reads from unmodified (C) sample
            methylated_reads: Reads from methylated (5mC) sample
            n_iterations: Number of Baum-Welch iterations
            filter_before_training: Whether to filter by score first

        Returns:
            Trained model
        """
        if n_iterations is None:
            n_iterations = self.config.max_iterations

        # Filter if requested
        if filter_before_training:
            control_reads = self.filter_by_score(control_reads)
            if methylated_reads:
                methylated_reads = self.filter_by_score(methylated_reads)

        # Combine datasets
        all_reads = control_reads.copy()
        if methylated_reads:
            all_reads.extend(methylated_reads)

        if len(all_reads) == 0:
            print("Warning: No reads passed filtering!")
            return self.model

        print(f"Training on {len(all_reads)} reads "
              f"({len(control_reads)} control"
              + (f", {len(methylated_reads)} methylated)" if methylated_reads else ")"))

        # Prepare tensor
        X = self.prepare_training_data(all_reads)

        # Run Baum-Welch using pomegranate's fit method
        self.model.fit(X)

        print(f"Training complete after {n_iterations} iterations")

        return self.model

    def train_iteratively(
        self,
        control_reads: List[SegmentedRead],
        methylated_reads: Optional[List[SegmentedRead]] = None,
        n_iterations: int = 10,
        log_interval: int = 1
    ) -> DenseHMM:
        """
        Train with manual iteration control and logging.

        Allows monitoring convergence and intermediate results.

        Args:
            control_reads: Control sample reads
            methylated_reads: Methylated sample reads
            n_iterations: Total iterations
            log_interval: Iterations between log messages

        Returns:
            Trained model
        """
        all_reads = control_reads.copy()
        if methylated_reads:
            all_reads.extend(methylated_reads)

        X = self.prepare_training_data(all_reads)

        print(f"Starting iterative training on {len(all_reads)} reads...")

        for i in range(n_iterations):
            # Single training step
            # Pomegranate's fit does multiple iterations internally,
            # so we use summarize + from_summaries for single steps
            self.model.summarize(X)
            self.model.from_summaries()

            # Compute log likelihood for monitoring
            with torch.no_grad():
                logp = self.model.log_probability(X).mean().item()
                self.training_history.append(logp)

            if (i + 1) % log_interval == 0:
                print(f"  Iteration {i+1}/{n_iterations}: "
                      f"mean log P = {logp:.2f}")

        print(f"Training complete. Final log P = {self.training_history[-1]:.2f}")

        return self.model

    def save_model(self, filepath: str) -> None:
        """
        Save trained model to file.

        Args:
            filepath: Output path (.pt for PyTorch format)
        """
        torch.save({
            'model_state': self.model.state_dict(),
            'config': self.config,
            'fork_indices': self.builder.fork_indices,
            'training_history': self.training_history
        }, filepath)
        print(f"Saved model to {filepath}")

    def load_model(self, filepath: str) -> DenseHMM:
        """
        Load trained model from file.

        Args:
            filepath: Path to saved model

        Returns:
            Loaded model
        """
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state'])
        self.training_history = checkpoint.get('training_history', [])
        print(f"Loaded model from {filepath}")
        return self.model


def train_from_tsvs(
    config: HMMConfig,
    control_tsv: str,
    methylated_tsv: Optional[str] = None,
    output_model_path: Optional[str] = None
) -> Tuple[DenseHMM, ProfileHMMBuilder]:
    """
    Convenience function for full training pipeline.

    Args:
        config: HMM configuration
        control_tsv: Path to control sample TSV
        methylated_tsv: Optional path to methylated sample TSV
        output_model_path: Optional path to save trained model

    Returns:
        (trained_model, builder) tuple
    """
    from .hmm_builder import build_hmm_from_config

    # Build initial model
    model, builder = build_hmm_from_config(config)

    # Load data
    loader = DataLoader(
        min_segments=config.min_segments_per_read,
        max_segments=config.max_segments_per_read
    )

    control_reads = loader.load_and_preprocess(
        control_tsv, label='control', normalize=True
    )

    methylated_reads = None
    if methylated_tsv:
        methylated_reads = loader.load_and_preprocess(
            methylated_tsv, label='5mC', normalize=True
        )

    # Train
    trainer = BaumWelchTrainer(model, builder, config)
    model = trainer.train(control_reads, methylated_reads)

    # Save if requested
    if output_model_path:
        trainer.save_model(output_model_path)

    return model, builder
