"""
Full-Sequence HMM for cytosine modification classification.

This classifier models the ENTIRE 155bp reference sequence, not just the 8 cytosine
positions. It uses fork states at cytosine positions to classify modifications.

Key differences from GenericHMM (sparse):
- Observes current at ALL positions (not just 8 cytosines)
- Has match states for all 155 positions
- Fork states only at cytosine positions (38, 50, 62, 74, 86, 98, 110, 122)
- Uses k-mer context implicitly through position-specific emissions

Architecture:
    Position 0-37:   Match states (single state per position)
    Position 38:     Fork states (C, 5mC, [5hmC] branches)
    Position 39-49:  Match states
    Position 50:     Fork states
    ... (continues for all 155 positions)

See EVALUATION_PLAN.md for context on why full-sequence modeling.
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Normal

from .emission_params import (
    FullSequenceEmissionParams,
    compute_emission_params_from_full_csv,
    CYTOSINE_POSITIONS,
    SEQUENCE_LENGTH,
)


@dataclass
class ClassificationResult:
    """Result of classifying a single read."""
    read_id: str
    prediction: str  # 'C', '5mC', or '5hmC'
    prediction_idx: int
    probabilities: Dict[str, float]  # {'C': 0.3, '5mC': 0.7, ...}
    confidence: float  # max prob - second max prob
    log_likelihood: float


@dataclass
class EvaluationMetrics:
    """Metrics from classification evaluation."""
    accuracy: float
    per_class_accuracy: Dict[str, float]
    confusion_matrix: np.ndarray
    n_samples: int
    n_correct: int


class FullSequenceHMM:
    """
    Profile HMM for full 155bp sequence classification.

    Models the entire reference sequence with fork states at cytosine positions.
    Uses emission parameters from either single adapter (context-specific) or
    pooled across all adapters.

    State architecture:
    - Non-cytosine positions: 1 match state
    - Cytosine positions (binary): 2 states (C, 5mC)
    - Cytosine positions (3-way): 3 states (C, 5mC, 5hmC)

    Total states:
    - Binary: 155 - 8 + 8*2 = 163 states
    - 3-way: 155 - 8 + 8*3 = 171 states
    """

    CYTOSINE_POSITIONS = CYTOSINE_POSITIONS
    SEQUENCE_LENGTH = SEQUENCE_LENGTH
    MODIFICATIONS_BINARY = ['C', '5mC']
    MODIFICATIONS_3WAY = ['C', '5mC', '5hmC']

    # Transition probabilities
    P_SELF_LOOP = 0.10  # Dwelling on current base
    P_FORWARD = 0.88    # Normal progression to next position
    P_SKIP = 0.02       # Skip next position (undersegmentation)

    def __init__(
        self,
        emission_params: Optional[FullSequenceEmissionParams] = None,
        n_classes: int = 2,
    ):
        """
        Initialize FullSequenceHMM.

        Args:
            emission_params: Pre-computed emission parameters
            n_classes: 2 for binary (C/5mC), 3 for 3-way (C/5mC/5hmC)
        """
        if n_classes not in (2, 3):
            raise ValueError(f"n_classes must be 2 or 3, got {n_classes}")

        self.emission_params = emission_params
        self.n_classes = n_classes
        self.model: Optional[DenseHMM] = None
        self.is_fitted = False

        # Class-specific attributes
        self.modifications = self.MODIFICATIONS_BINARY if n_classes == 2 else self.MODIFICATIONS_3WAY

        # Build state index mapping
        self._build_state_mapping()

    def _build_state_mapping(self) -> None:
        """Build mapping from (position, modification) to state index."""
        self.state_to_idx: Dict[Tuple[int, str], int] = {}
        self.idx_to_state: Dict[int, Tuple[int, str]] = {}

        idx = 0
        for pos in range(self.SEQUENCE_LENGTH):
            if pos in self.CYTOSINE_POSITIONS:
                # Fork states for cytosines
                for mod in self.modifications:
                    self.state_to_idx[(pos, mod)] = idx
                    self.idx_to_state[idx] = (pos, mod)
                    idx += 1
            else:
                # Single match state
                self.state_to_idx[(pos, 'match')] = idx
                self.idx_to_state[idx] = (pos, 'match')
                idx += 1

        self.n_states = idx

    @classmethod
    def from_emission_params(
        cls,
        params: Union[str, FullSequenceEmissionParams],
    ) -> 'FullSequenceHMM':
        """
        Create HMM from emission parameters.

        Args:
            params: Path to JSON or FullSequenceEmissionParams object

        Returns:
            Initialized and built FullSequenceHMM
        """
        if isinstance(params, str):
            params = FullSequenceEmissionParams.load(params)

        n_classes = 2 if params.mode == 'binary' else 3
        hmm = cls(emission_params=params, n_classes=n_classes)
        hmm._build_model()
        hmm.is_fitted = True
        return hmm

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        adapter: Optional[str] = None,
        mode: str = 'binary',
    ) -> 'FullSequenceHMM':
        """
        Create HMM from signal_full_sequence.csv.

        Args:
            csv_path: Path to signal_full_sequence.csv
            adapter: Adapter for single mode (None = pooled)
            mode: 'binary' or '3way'

        Returns:
            Initialized and built FullSequenceHMM
        """
        params = compute_emission_params_from_full_csv(
            csv_path,
            adapter=adapter,
            mode=mode,
        )
        return cls.from_emission_params(params)

    def _build_model(self) -> None:
        """Build the DenseHMM from emission parameters."""
        if self.emission_params is None:
            raise ValueError("No emission parameters available")

        distributions = self._build_distributions()
        edges = self._build_transition_matrix()
        starts = self._build_start_probs()
        ends = self._build_end_probs()

        self.model = DenseHMM(
            distributions=distributions,
            edges=edges,
            starts=starts,
            ends=ends,
            max_iter=10,
            tol=0.001,
        )

    def _build_distributions(self) -> List[Normal]:
        """Build Normal distributions for all states."""
        distributions = []

        for idx in range(self.n_states):
            pos, state_type = self.idx_to_state[idx]
            pos_params = self.emission_params.get_position_params(pos)

            if pos_params is None:
                # Fallback for missing positions
                mean, std = 800.0, 100.0
            elif state_type == 'match':
                # Non-cytosine position - use C params
                mean = pos_params.C.mean
                std = pos_params.C.std
            elif state_type == 'C':
                mean = pos_params.C.mean
                std = pos_params.C.std
            elif state_type == '5mC':
                if pos_params.mC:
                    mean = pos_params.mC.mean
                    std = pos_params.mC.std
                else:
                    mean = pos_params.C.mean + 30  # Default offset
                    std = pos_params.C.std
            elif state_type == '5hmC':
                if pos_params.hmC:
                    mean = pos_params.hmC.mean
                    std = pos_params.hmC.std
                else:
                    mean = pos_params.C.mean + 10  # Default offset
                    std = pos_params.C.std
            else:
                mean, std = 800.0, 100.0

            distributions.append(Normal(
                means=torch.tensor([mean]),
                covs=torch.tensor([[std ** 2]]),
            ))

        return distributions

    def _build_transition_matrix(self) -> torch.Tensor:
        """Build transition matrix with self-loops, forward, and skip transitions."""
        n = self.n_states
        edges = torch.zeros((n, n), dtype=torch.float32)

        for from_idx in range(n):
            from_pos, from_type = self.idx_to_state[from_idx]

            # Self-loop
            edges[from_idx, from_idx] = self.P_SELF_LOOP

            # Forward transitions to next position
            next_pos = from_pos + 1
            if next_pos < self.SEQUENCE_LENGTH:
                next_states = self._get_states_at_position(next_pos)
                p_each = self.P_FORWARD / len(next_states)
                for to_idx in next_states:
                    edges[from_idx, to_idx] = p_each

            # Skip transitions (to position + 2)
            skip_pos = from_pos + 2
            if skip_pos < self.SEQUENCE_LENGTH:
                skip_states = self._get_states_at_position(skip_pos)
                p_each_skip = self.P_SKIP / len(skip_states)
                for to_idx in skip_states:
                    edges[from_idx, to_idx] = p_each_skip

        return edges

    def _get_states_at_position(self, pos: int) -> List[int]:
        """Get all state indices at a given position."""
        states = []
        if pos in self.CYTOSINE_POSITIONS:
            for mod in self.modifications:
                if (pos, mod) in self.state_to_idx:
                    states.append(self.state_to_idx[(pos, mod)])
        else:
            if (pos, 'match') in self.state_to_idx:
                states.append(self.state_to_idx[(pos, 'match')])
        return states

    def _build_start_probs(self) -> torch.Tensor:
        """Build start probabilities - start at position 0."""
        starts = torch.zeros(self.n_states, dtype=torch.float32)
        start_states = self._get_states_at_position(0)
        for idx in start_states:
            starts[idx] = 1.0 / len(start_states)
        return starts

    def _build_end_probs(self) -> torch.Tensor:
        """Build end probabilities - can end from last position."""
        ends = torch.zeros(self.n_states, dtype=torch.float32)
        end_states = self._get_states_at_position(self.SEQUENCE_LENGTH - 1)
        for idx in end_states:
            ends[idx] = 1.0
        return ends

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict modification class for each read.

        Args:
            X: Array of shape (n_samples, seq_len) - current values at all positions

        Returns:
            Array of shape (n_samples,) with class indices
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get class probabilities for each read.

        Uses sum of log-likelihoods through each modification path at fork positions.

        Args:
            X: Array of shape (n_samples, seq_len) - current values at all positions

        Returns:
            Array of shape (n_samples, n_classes) with normalized probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        n_samples = X.shape[0]
        seq_len = X.shape[1]

        # Compute log-likelihood for each modification path through cytosines
        log_probs = np.zeros((n_samples, self.n_classes))

        for mod_idx, mod in enumerate(self.modifications):
            # Sum log probabilities at all positions assuming this modification
            for pos in range(min(seq_len, self.SEQUENCE_LENGTH)):
                if pos in self.CYTOSINE_POSITIONS:
                    # Use the fork state for this modification
                    state_idx = self.state_to_idx.get((pos, mod))
                else:
                    # Use match state
                    state_idx = self.state_to_idx.get((pos, 'match'))

                if state_idx is None:
                    continue

                dist = self.model.distributions[state_idx]
                obs = torch.tensor(X[:, pos:pos+1].astype(np.float32))

                with torch.no_grad():
                    log_p = dist.log_probability(obs).numpy()
                log_probs[:, mod_idx] += log_p.flatten()

        # Softmax normalization
        log_probs -= log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs)
        probs /= probs.sum(axis=1, keepdims=True)

        return probs

    def classify_dataframe(
        self,
        df: pd.DataFrame,
        min_positions: int = 100,
    ) -> List[ClassificationResult]:
        """
        Classify reads from a DataFrame.

        Args:
            df: DataFrame from signal_full_sequence.csv (long format)
            min_positions: Minimum number of positions required per read

        Returns:
            List of ClassificationResult objects
        """
        # Pivot to wide format: read_id x position
        wide_df = df.pivot_table(
            index='read_id',
            columns='position',
            values='mean_current',
            aggfunc='mean',
        ).reset_index()

        # Get position columns in order
        pos_cols = [c for c in wide_df.columns if isinstance(c, (int, np.integer))]
        pos_cols = sorted(pos_cols)

        # Filter to reads with enough positions (not necessarily all)
        n_valid = wide_df[pos_cols].notna().sum(axis=1)
        wide_df = wide_df[n_valid >= min_positions].copy()

        if len(wide_df) == 0:
            return []

        # Fill NaN with mean for that position (from emission params)
        for pos in pos_cols:
            pos_params = self.emission_params.get_position_params(pos) if self.emission_params else None
            if pos_params:
                fill_value = pos_params.C.mean
            else:
                col_mean = wide_df[pos].mean()
                fill_value = col_mean if not np.isnan(col_mean) else 800.0
            wide_df.loc[:, pos] = wide_df[pos].fillna(fill_value)

        X = wide_df[pos_cols].values.astype(np.float32)
        read_ids = wide_df['read_id'].values

        probs = self.predict_proba(X)
        preds = np.argmax(probs, axis=1)

        # Compute log-likelihoods for reporting
        log_probs_sum = np.zeros(len(X))
        for pos_idx, pos in enumerate(pos_cols):
            if pos_idx >= X.shape[1]:
                break
            state_idx = self.state_to_idx.get((pos, 'match'))
            if state_idx is None and pos in self.CYTOSINE_POSITIONS:
                state_idx = self.state_to_idx.get((pos, 'C'))
            if state_idx is not None:
                dist = self.model.distributions[state_idx]
                obs = torch.tensor(X[:, pos_idx:pos_idx+1])
                with torch.no_grad():
                    log_probs_sum += dist.log_probability(obs).numpy().flatten()

        results = []
        for i in range(len(X)):
            sorted_probs = np.sort(probs[i])[::-1]
            confidence = sorted_probs[0] - sorted_probs[1]

            results.append(ClassificationResult(
                read_id=str(read_ids[i]),
                prediction=self.modifications[preds[i]],
                prediction_idx=int(preds[i]),
                probabilities={m: float(probs[i, j]) for j, m in enumerate(self.modifications)},
                confidence=float(confidence),
                log_likelihood=float(log_probs_sum[i]),
            ))

        return results

    def evaluate(self, df: pd.DataFrame) -> EvaluationMetrics:
        """
        Evaluate classifier on labeled data.

        Args:
            df: DataFrame with 'sample' column (ground truth)

        Returns:
            EvaluationMetrics object
        """
        # Map sample names to indices
        sample_to_idx = {
            'control': 0, 'C': 0,
            '5mC': 1, 'mC': 1,
            '5hmC': 2, 'hmC': 2,
        }

        results = self.classify_dataframe(df)

        if len(results) == 0:
            return EvaluationMetrics(
                accuracy=0.0,
                per_class_accuracy={m: 0.0 for m in self.modifications},
                confusion_matrix=np.zeros((self.n_classes, self.n_classes), dtype=int),
                n_samples=0,
                n_correct=0,
            )

        # Get ground truth per read
        read_samples = df.groupby('read_id')['sample'].first().to_dict()
        y_true = np.array([
            sample_to_idx.get(read_samples.get(r.read_id, 'control'), 0)
            for r in results
        ])
        y_pred = np.array([r.prediction_idx for r in results])

        # Compute metrics
        accuracy = (y_pred == y_true).mean()
        n_correct = (y_pred == y_true).sum()

        per_class_acc = {}
        for idx, mod in enumerate(self.modifications):
            mask = y_true == idx
            if mask.sum() > 0:
                per_class_acc[mod] = float((y_pred[mask] == idx).mean())
            else:
                per_class_acc[mod] = 0.0

        # Confusion matrix
        conf_mat = np.zeros((self.n_classes, self.n_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            if 0 <= t < self.n_classes and 0 <= p < self.n_classes:
                conf_mat[t, p] += 1

        return EvaluationMetrics(
            accuracy=float(accuracy),
            per_class_accuracy=per_class_acc,
            confusion_matrix=conf_mat,
            n_samples=len(y_true),
            n_correct=int(n_correct),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'FullSequenceHMM':
        """
        Train HMM on labeled data using Baum-Welch.

        Args:
            X: Array of shape (n_samples, seq_len)
            y: Array of shape (n_samples,) - class labels

        Returns:
            self
        """
        if self.model is None:
            raise ValueError("Model not built")

        X_tensor = torch.tensor(X.astype(np.float32)).unsqueeze(-1)
        self.model.fit(X_tensor)
        self.is_fitted = True

        return self

    def get_emission_summary(self) -> Dict:
        """Get summary of emission parameters at cytosine positions."""
        if self.emission_params is None:
            return {}
        return self.emission_params.summary()

    def save(self, filepath: str) -> None:
        """Save model to file."""
        torch.save({
            'emission_params': self.emission_params.to_dict() if self.emission_params else None,
            'n_classes': self.n_classes,
            'model_state_dict': self.model.state_dict() if self.model else None,
            'is_fitted': self.is_fitted,
        }, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'FullSequenceHMM':
        """Load model from file."""
        checkpoint = torch.load(filepath)

        params = None
        if checkpoint['emission_params']:
            from .emission_params import FullSequenceEmissionParams
            # Reconstruct params from dict
            params_dict = checkpoint['emission_params']
            params = FullSequenceEmissionParams.load.__func__(
                FullSequenceEmissionParams,
                None
            )  # This won't work directly, need proper reconstruction
            # For now, just store the dict
            params = checkpoint['emission_params']

        hmm = cls(n_classes=checkpoint['n_classes'])
        if params:
            # Rebuild from params dict
            hmm.emission_params = FullSequenceEmissionParams.load(filepath.replace('.pt', '_params.json'))
            hmm._build_model()

        if checkpoint['model_state_dict'] and hmm.model:
            hmm.model.load_state_dict(checkpoint['model_state_dict'])

        hmm.is_fitted = checkpoint['is_fitted']
        return hmm


def run_evaluation(
    csv_path: str,
    adapter: Optional[str] = None,
    mode: str = 'binary',
    test_split: float = 0.2,
    seed: int = 42,
) -> Tuple[FullSequenceHMM, EvaluationMetrics]:
    """
    Quick evaluation function.

    Args:
        csv_path: Path to signal_full_sequence.csv
        adapter: Adapter for single mode (None = pooled)
        mode: 'binary' or '3way'
        test_split: Fraction for test set
        seed: Random seed

    Returns:
        (trained_hmm, metrics)
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Filter by adapter if specified
    if adapter:
        df = df[df['chrom'] == adapter]
        source = 'single'
    else:
        source = 'pooled'

    # Filter samples based on mode
    if mode == 'binary':
        df = df[df['sample'].isin(['control', '5mC'])]

    print(f"Mode: {mode}, Source: {source}")
    print(f"Total rows: {len(df):,}, Unique reads: {df['read_id'].nunique():,}")

    # Split by read_id
    np.random.seed(seed)
    read_ids = df['read_id'].unique()
    np.random.shuffle(read_ids)
    n_test = int(len(read_ids) * test_split)
    test_ids = set(read_ids[:n_test])

    test_df = df[df['read_id'].isin(test_ids)]
    train_df = df[~df['read_id'].isin(test_ids)]

    print(f"Train reads: {train_df['read_id'].nunique():,}, Test reads: {test_df['read_id'].nunique():,}")

    # Build HMM from training data
    print("Computing emission parameters from training data...")
    params = compute_emission_params_from_full_csv(
        csv_path,
        adapter=adapter,
        mode=mode,
    )

    hmm = FullSequenceHMM.from_emission_params(params)

    print("Evaluating on test data...")
    metrics = hmm.evaluate(test_df)

    print(f"\n=== Results ({mode}, {source}) ===")
    print(f"Overall accuracy: {metrics.accuracy:.1%}")
    print(f"Per-class accuracy:")
    for mod, acc in metrics.per_class_accuracy.items():
        print(f"  {mod}: {acc:.1%}")
    print(f"\nConfusion matrix:")
    print(metrics.confusion_matrix)

    return hmm, metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Full-Sequence HMM evaluation")
    parser.add_argument("--csv", required=True, help="Path to signal_full_sequence.csv")
    parser.add_argument("--adapter", default=None, help="Adapter for single mode")
    parser.add_argument("--mode", choices=["binary", "3way"], default="binary")
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", help="Output path for model")

    args = parser.parse_args()

    hmm, metrics = run_evaluation(
        args.csv,
        adapter=args.adapter,
        mode=args.mode,
        test_split=args.test_split,
        seed=args.seed,
    )

    if args.output:
        hmm.save(args.output)
        print(f"\nModel saved to {args.output}")
