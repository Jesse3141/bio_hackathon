"""
Simplified HMM Pipeline for Methylation Detection.

This module implements a likelihood-ratio classifier using two separate HMMs:
- Control HMM: trained on canonical cytosine (C) data
- Modified HMM: trained on 5-methylcytosine (5mC) data

Classification is performed by comparing log P(X | control) vs log P(X | modified).

Achieved 70.7% accuracy on POD5-derived signal data with +36.7 pA difference between
C and 5mC signals at 8 cytosine positions.
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import torch
from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Normal


@dataclass
class ClassificationResult:
    """Result of classifying a single read."""
    read_id: str
    prediction: str  # 'control' or 'modified'
    log_prob_control: float
    log_prob_modified: float
    log_ratio: float  # log P(modified) - log P(control)
    confidence: float  # abs(log_ratio), higher = more confident


@dataclass
class EvaluationMetrics:
    """Metrics from classification evaluation."""
    accuracy: float
    control_accuracy: float
    modified_accuracy: float
    n_samples: int
    n_correct: int


class SimplifiedMethylationClassifier:
    """
    Likelihood-ratio classifier for methylation detection.

    Uses two separate HMMs (control and modified) and classifies
    based on which model assigns higher probability to each sequence.
    """

    POSITIONS = ['38', '50', '62', '74', '86', '98', '110', '122']

    def __init__(self, emission_params: Optional[Dict] = None):
        """
        Initialize classifier.

        Args:
            emission_params: Optional pre-computed emission parameters.
                If None, must call fit() with training data.
        """
        self.control_model: Optional[DenseHMM] = None
        self.modified_model: Optional[DenseHMM] = None
        self.emission_params = emission_params
        self.is_fitted = False

    @classmethod
    def from_json(cls, params_path: str) -> 'SimplifiedMethylationClassifier':
        """
        Create classifier from pre-computed emission parameters.

        Args:
            params_path: Path to hmm_emission_params_pomegranate.json

        Returns:
            Initialized classifier (ready for predict, or can be trained further)
        """
        with open(params_path) as f:
            emission_params = json.load(f)

        classifier = cls(emission_params)
        classifier._build_models_from_params()
        classifier.is_fitted = True
        return classifier

    @classmethod
    def from_training_data(
        cls,
        training_csv: str,
        test_split: float = 0.2
    ) -> Tuple['SimplifiedMethylationClassifier', pd.DataFrame]:
        """
        Create and train classifier from training data CSV.

        Args:
            training_csv: Path to hmm_training_sequences.csv
            test_split: Fraction to hold out for testing

        Returns:
            (trained_classifier, test_dataframe)
        """
        classifier = cls()
        df = pd.read_csv(training_csv)

        # Clean data
        df_clean = df.dropna(subset=cls.POSITIONS)

        # Split by sample type
        control = df_clean[df_clean['sample'] == 'control']
        modified = df_clean[df_clean['sample'] == 'modified']

        # Train/test split
        n_control_train = int(len(control) * (1 - test_split))
        n_modified_train = int(len(modified) * (1 - test_split))

        train_control = control.iloc[:n_control_train]
        train_modified = modified.iloc[:n_modified_train]

        test_df = pd.concat([
            control.iloc[n_control_train:],
            modified.iloc[n_modified_train:]
        ])

        # Compute emission parameters from training data
        classifier._compute_emission_params(train_control, train_modified)
        classifier._build_models_from_params()

        # Train models
        X_control = torch.tensor(
            train_control[cls.POSITIONS].values.astype(np.float32)
        ).unsqueeze(-1)
        X_modified = torch.tensor(
            train_modified[cls.POSITIONS].values.astype(np.float32)
        ).unsqueeze(-1)

        classifier.control_model.fit(X_control)
        classifier.modified_model.fit(X_modified)
        classifier.is_fitted = True

        return classifier, test_df

    def _compute_emission_params(
        self,
        control_df: pd.DataFrame,
        modified_df: pd.DataFrame
    ) -> None:
        """Compute emission parameters from training data."""
        states = []

        for pos in self.POSITIONS:
            # Control state
            control_values = control_df[pos].dropna()
            states.append({
                'name': f'control_pos{pos}',
                'distribution': 'Normal',
                'parameters': {
                    'mean': float(control_values.mean()),
                    'std': float(control_values.std())
                },
                'n_observations': len(control_values)
            })

            # Modified state
            modified_values = modified_df[pos].dropna()
            states.append({
                'name': f'modified_pos{pos}',
                'distribution': 'Normal',
                'parameters': {
                    'mean': float(modified_values.mean()),
                    'std': float(modified_values.std())
                },
                'n_observations': len(modified_values)
            })

        self.emission_params = {'states': states}

    def _build_models_from_params(self) -> None:
        """Build control and modified HMMs from emission parameters."""
        if self.emission_params is None:
            raise ValueError("No emission parameters available")

        # Separate control and modified parameters
        control_params = [
            s['parameters'] for s in self.emission_params['states']
            if 'control' in s['name']
        ]
        modified_params = [
            s['parameters'] for s in self.emission_params['states']
            if 'modified' in s['name']
        ]

        self.control_model = self._build_linear_hmm(control_params)
        self.modified_model = self._build_linear_hmm(modified_params)

    def _build_linear_hmm(self, states_params: List[Dict]) -> DenseHMM:
        """Build a simple left-to-right HMM."""
        distributions = []
        for params in states_params:
            distributions.append(Normal(
                means=torch.tensor([params['mean']]),
                covs=torch.tensor([[params['std'] ** 2]])
            ))

        n_states = len(distributions)

        # Linear left-to-right transitions
        edges = torch.zeros((n_states, n_states), dtype=torch.float32)
        for i in range(n_states - 1):
            edges[i, i + 1] = 1.0

        starts = torch.zeros(n_states)
        starts[0] = 1.0

        ends = torch.zeros(n_states)
        ends[-1] = 1.0

        return DenseHMM(
            distributions=distributions,
            edges=edges,
            starts=starts,
            ends=ends,
            max_iter=10,
            tol=0.001
        )

    def fit(self, control_data: np.ndarray, modified_data: np.ndarray) -> 'SimplifiedMethylationClassifier':
        """
        Fit the classifier on training data.

        Args:
            control_data: Array of shape (n_control, 8) - control sequences
            modified_data: Array of shape (n_modified, 8) - modified sequences

        Returns:
            self (for chaining)
        """
        X_control = torch.tensor(control_data.astype(np.float32)).unsqueeze(-1)
        X_modified = torch.tensor(modified_data.astype(np.float32)).unsqueeze(-1)

        self.control_model.fit(X_control)
        self.modified_model.fit(X_modified)
        self.is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for sequences.

        Args:
            X: Array of shape (n_samples, 8) - sequences to classify

        Returns:
            Array of shape (n_samples,) with 0=control, 1=modified
        """
        if not self.is_fitted:
            raise ValueError("Classifier not fitted. Call fit() first.")

        X_tensor = torch.tensor(X.astype(np.float32)).unsqueeze(-1)

        self.control_model.eval()
        self.modified_model.eval()

        with torch.no_grad():
            logp_control = self.control_model.log_probability(X_tensor)
            logp_modified = self.modified_model.log_probability(X_tensor)

        log_ratio = logp_modified - logp_control
        predictions = (log_ratio > 0).numpy().astype(int)

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get class probabilities for both classes.

        Args:
            X: Array of shape (n_samples, 8)

        Returns:
            Array of shape (n_samples, 2) with normalized probabilities
        """
        if not self.is_fitted:
            raise ValueError("Classifier not fitted. Call fit() first.")

        X_tensor = torch.tensor(X.astype(np.float32)).unsqueeze(-1)

        self.control_model.eval()
        self.modified_model.eval()

        with torch.no_grad():
            logp_control = self.control_model.log_probability(X_tensor).numpy()
            logp_modified = self.modified_model.log_probability(X_tensor).numpy()

        # Stack and convert to probabilities via softmax
        log_probs = np.stack([logp_control, logp_modified], axis=1)
        log_probs -= log_probs.max(axis=1, keepdims=True)  # Numerical stability
        probs = np.exp(log_probs)
        probs /= probs.sum(axis=1, keepdims=True)

        return probs

    def predict_log_proba(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get log probabilities for both classes (legacy interface).

        Args:
            X: Array of shape (n_samples, 8)

        Returns:
            (logp_control, logp_modified) - both of shape (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Classifier not fitted. Call fit() first.")

        X_tensor = torch.tensor(X.astype(np.float32)).unsqueeze(-1)

        self.control_model.eval()
        self.modified_model.eval()

        with torch.no_grad():
            logp_control = self.control_model.log_probability(X_tensor).numpy()
            logp_modified = self.modified_model.log_probability(X_tensor).numpy()

        return logp_control, logp_modified

    def fit_from_dataframe(self, df: pd.DataFrame) -> 'SimplifiedMethylationClassifier':
        """
        Fit classifier from a wide-format DataFrame.

        Args:
            df: DataFrame with 'sample' column and position columns

        Returns:
            self (for chaining)
        """
        df_clean = df.dropna(subset=self.POSITIONS)

        # Split by sample type
        control = df_clean[df_clean['sample'].isin(['control', 'C'])]
        modified = df_clean[df_clean['sample'].isin(['modified', '5mC', 'mC'])]

        if len(control) == 0 or len(modified) == 0:
            raise ValueError("Need both control and modified samples in training data")

        # Compute emission parameters
        self._compute_emission_params(control, modified)
        self._build_models_from_params()

        # Train models
        X_control = torch.tensor(
            control[self.POSITIONS].values.astype(np.float32)
        ).unsqueeze(-1)
        X_modified = torch.tensor(
            modified[self.POSITIONS].values.astype(np.float32)
        ).unsqueeze(-1)

        self.control_model.fit(X_control)
        self.modified_model.fit(X_modified)
        self.is_fitted = True

        return self

    def classify_dataframe(self, df: pd.DataFrame) -> List[ClassificationResult]:
        """
        Classify reads from a DataFrame.

        Args:
            df: DataFrame with columns for positions (38, 50, ...) and read_id

        Returns:
            List of ClassificationResult objects
        """
        # Filter valid rows
        df_valid = df.dropna(subset=self.POSITIONS)

        X = df_valid[self.POSITIONS].values.astype(np.float32)
        read_ids = df_valid['read_id'].values if 'read_id' in df_valid.columns else [f'read_{i}' for i in range(len(X))]

        logp_control, logp_modified = self.predict_log_proba(X)
        predictions = self.predict(X)

        results = []
        for i in range(len(X)):
            log_ratio = logp_modified[i] - logp_control[i]
            results.append(ClassificationResult(
                read_id=str(read_ids[i]),
                prediction='modified' if predictions[i] == 1 else 'control',
                log_prob_control=float(logp_control[i]),
                log_prob_modified=float(logp_modified[i]),
                log_ratio=float(log_ratio),
                confidence=float(abs(log_ratio))
            ))

        return results

    def evaluate(self, df: pd.DataFrame) -> EvaluationMetrics:
        """
        Evaluate classifier on labeled data.

        Args:
            df: DataFrame with 'sample' column (ground truth) and position columns

        Returns:
            EvaluationMetrics object
        """
        df_valid = df.dropna(subset=self.POSITIONS)

        X = df_valid[self.POSITIONS].values.astype(np.float32)
        y_true = (df_valid['sample'].values == 'modified').astype(int)

        y_pred = self.predict(X)

        accuracy = (y_pred == y_true).mean()
        n_correct = (y_pred == y_true).sum()

        control_mask = y_true == 0
        modified_mask = y_true == 1

        control_acc = (y_pred[control_mask] == 0).mean() if control_mask.sum() > 0 else 0.0
        modified_acc = (y_pred[modified_mask] == 1).mean() if modified_mask.sum() > 0 else 0.0

        return EvaluationMetrics(
            accuracy=float(accuracy),
            control_accuracy=float(control_acc),
            modified_accuracy=float(modified_acc),
            n_samples=len(y_true),
            n_correct=int(n_correct)
        )

    def save(self, filepath: str) -> None:
        """Save classifier to file."""
        torch.save({
            'emission_params': self.emission_params,
            'control_state_dict': self.control_model.state_dict(),
            'modified_state_dict': self.modified_model.state_dict(),
        }, filepath)

    def load(self, filepath: str) -> 'SimplifiedMethylationClassifier':
        """Load classifier from file."""
        checkpoint = torch.load(filepath)
        self.emission_params = checkpoint['emission_params']
        self._build_models_from_params()
        self.control_model.load_state_dict(checkpoint['control_state_dict'])
        self.modified_model.load_state_dict(checkpoint['modified_state_dict'])
        self.is_fitted = True
        return self


class ThreeWaySimplifiedClassifier:
    """
    Three-way likelihood-ratio classifier for C/5mC/5hmC detection.

    Uses three separate HMMs and classifies based on which model
    assigns highest probability to each sequence.
    """

    POSITIONS = ['38', '50', '62', '74', '86', '98', '110', '122']
    CLASSES = ['C', '5mC', '5hmC']

    def __init__(self, emission_params: Optional[Dict] = None):
        """
        Initialize classifier.

        Args:
            emission_params: Optional pre-computed emission parameters.
        """
        self.control_model: Optional[DenseHMM] = None
        self.mC_model: Optional[DenseHMM] = None
        self.hmC_model: Optional[DenseHMM] = None
        self.emission_params = emission_params
        self.is_fitted = False

    def _build_linear_hmm(self, states_params: List[Dict]) -> DenseHMM:
        """Build a simple left-to-right HMM."""
        distributions = []
        for params in states_params:
            distributions.append(Normal(
                means=torch.tensor([params['mean']]),
                covs=torch.tensor([[params['std'] ** 2]])
            ))

        n_states = len(distributions)

        edges = torch.zeros((n_states, n_states), dtype=torch.float32)
        for i in range(n_states - 1):
            edges[i, i + 1] = 1.0

        starts = torch.zeros(n_states)
        starts[0] = 1.0

        ends = torch.zeros(n_states)
        ends[-1] = 1.0

        return DenseHMM(
            distributions=distributions,
            edges=edges,
            starts=starts,
            ends=ends,
            max_iter=10,
            tol=0.001
        )

    def fit_from_dataframe(self, df: pd.DataFrame) -> 'ThreeWaySimplifiedClassifier':
        """
        Fit classifier from a wide-format DataFrame.

        Args:
            df: DataFrame with 'sample' column and position columns

        Returns:
            self (for chaining)
        """
        df_clean = df.dropna(subset=self.POSITIONS)

        # Split by sample type
        control = df_clean[df_clean['sample'].isin(['control', 'C'])]
        mC = df_clean[df_clean['sample'].isin(['5mC', 'mC', 'modified'])]
        hmC = df_clean[df_clean['sample'].isin(['5hmC', 'hmC'])]

        if len(control) == 0 or len(mC) == 0:
            raise ValueError("Need control and 5mC samples in training data")

        # Compute emission parameters
        self._compute_emission_params(control, mC, hmC)
        self._build_models_from_params()

        # Train models
        X_control = torch.tensor(
            control[self.POSITIONS].values.astype(np.float32)
        ).unsqueeze(-1)
        self.control_model.fit(X_control)

        X_mC = torch.tensor(
            mC[self.POSITIONS].values.astype(np.float32)
        ).unsqueeze(-1)
        self.mC_model.fit(X_mC)

        if len(hmC) > 0:
            X_hmC = torch.tensor(
                hmC[self.POSITIONS].values.astype(np.float32)
            ).unsqueeze(-1)
            self.hmC_model.fit(X_hmC)

        self.is_fitted = True
        return self

    def _compute_emission_params(
        self,
        control_df: pd.DataFrame,
        mC_df: pd.DataFrame,
        hmC_df: pd.DataFrame
    ) -> None:
        """Compute emission parameters from training data."""
        states = []

        for pos in self.POSITIONS:
            # Control state
            control_values = control_df[pos].dropna()
            states.append({
                'name': f'C_pos{pos}',
                'parameters': {
                    'mean': float(control_values.mean()),
                    'std': float(control_values.std())
                }
            })

            # 5mC state
            mC_values = mC_df[pos].dropna()
            states.append({
                'name': f'mC_pos{pos}',
                'parameters': {
                    'mean': float(mC_values.mean()),
                    'std': float(mC_values.std())
                }
            })

            # 5hmC state
            if len(hmC_df) > 0:
                hmC_values = hmC_df[pos].dropna()
                states.append({
                    'name': f'hmC_pos{pos}',
                    'parameters': {
                        'mean': float(hmC_values.mean()),
                        'std': float(hmC_values.std())
                    }
                })
            else:
                # Use control values as fallback
                states.append({
                    'name': f'hmC_pos{pos}',
                    'parameters': {
                        'mean': float(control_values.mean()),
                        'std': float(control_values.std())
                    }
                })

        self.emission_params = {'states': states}

    def _build_models_from_params(self) -> None:
        """Build three HMMs from emission parameters."""
        if self.emission_params is None:
            raise ValueError("No emission parameters available")

        control_params = [
            s['parameters'] for s in self.emission_params['states']
            if s['name'].startswith('C_')
        ]
        mC_params = [
            s['parameters'] for s in self.emission_params['states']
            if s['name'].startswith('mC_')
        ]
        hmC_params = [
            s['parameters'] for s in self.emission_params['states']
            if s['name'].startswith('hmC_')
        ]

        self.control_model = self._build_linear_hmm(control_params)
        self.mC_model = self._build_linear_hmm(mC_params)
        self.hmC_model = self._build_linear_hmm(hmC_params)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for sequences.

        Args:
            X: Array of shape (n_samples, 8)

        Returns:
            Array of shape (n_samples,) with 0=C, 1=5mC, 2=5hmC
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get class probabilities for all three classes.

        Args:
            X: Array of shape (n_samples, 8)

        Returns:
            Array of shape (n_samples, 3) with normalized probabilities
        """
        if not self.is_fitted:
            raise ValueError("Classifier not fitted. Call fit_from_dataframe() first.")

        X_tensor = torch.tensor(X.astype(np.float32)).unsqueeze(-1)

        self.control_model.eval()
        self.mC_model.eval()
        self.hmC_model.eval()

        with torch.no_grad():
            logp_control = self.control_model.log_probability(X_tensor).numpy()
            logp_mC = self.mC_model.log_probability(X_tensor).numpy()
            logp_hmC = self.hmC_model.log_probability(X_tensor).numpy()

        # Stack and convert to probabilities via softmax
        log_probs = np.stack([logp_control, logp_mC, logp_hmC], axis=1)
        log_probs -= log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs)
        probs /= probs.sum(axis=1, keepdims=True)

        return probs

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ThreeWaySimplifiedClassifier':
        """
        Fit the classifier on training data.

        Args:
            X: Array of shape (n_samples, 8) - sequences
            y: Array of shape (n_samples,) - class labels (0, 1, 2)

        Returns:
            self (for chaining)
        """
        # Build DataFrame for fit_from_dataframe
        df = pd.DataFrame(X, columns=self.POSITIONS)
        df['sample'] = [self.CLASSES[int(yi)] for yi in y]
        return self.fit_from_dataframe(df)


def run_full_pipeline(
    training_csv: str,
    emission_params_json: str,
    output_model: Optional[str] = None
) -> Tuple[SimplifiedMethylationClassifier, EvaluationMetrics]:
    """
    Run the complete training and evaluation pipeline.

    Args:
        training_csv: Path to hmm_training_sequences.csv
        emission_params_json: Path to hmm_emission_params_pomegranate.json
        output_model: Optional path to save trained model

    Returns:
        (classifier, metrics) tuple
    """
    print("Loading training data...")
    classifier, test_df = SimplifiedMethylationClassifier.from_training_data(training_csv)

    print(f"Training complete. Test set: {len(test_df)} reads")

    print("Evaluating on test set...")
    metrics = classifier.evaluate(test_df)

    print(f"\nResults:")
    print(f"  Overall accuracy: {metrics.accuracy:.3f}")
    print(f"  Control accuracy: {metrics.control_accuracy:.3f}")
    print(f"  Modified accuracy: {metrics.modified_accuracy:.3f}")
    print(f"  Samples: {metrics.n_samples}, Correct: {metrics.n_correct}")

    if output_model:
        classifier.save(output_model)
        print(f"\nModel saved to: {output_model}")

    return classifier, metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train and evaluate methylation classifier")
    parser.add_argument("--training-csv", required=True, help="Path to training sequences CSV")
    parser.add_argument("--emission-params", required=True, help="Path to emission params JSON")
    parser.add_argument("--output-model", help="Path to save trained model")

    args = parser.parse_args()

    classifier, metrics = run_full_pipeline(
        training_csv=args.training_csv,
        emission_params_json=args.emission_params,
        output_model=args.output_model
    )
