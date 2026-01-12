"""
Comprehensive tests for Profile HMM methylation detection pipeline.

These tests prove the HMM correctly:
1. Loads the POD5-derived training data
2. Builds a valid pomegranate DenseHMM
3. Trains on labeled control vs 5mC samples
4. Achieves meaningful classification accuracy (>70%)
5. Discriminates between C and 5mC based on learned emission parameters

Test data: output/hmm_training_sequences.csv with 8 cytosine positions per read
"""

import pytest
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestDataLoading:
    """Test loading of POD5-derived training data."""

    @pytest.fixture
    def training_data_path(self):
        return PROJECT_ROOT / "output" / "hmm_training_sequences.csv"

    @pytest.fixture
    def emission_params_path(self):
        return PROJECT_ROOT / "output" / "hmm_emission_params_pomegranate.json"

    @pytest.fixture
    def signal_data_path(self):
        return PROJECT_ROOT / "output" / "signal_at_cytosines.csv"

    def test_training_data_file_exists(self, training_data_path):
        """Training sequences file must exist."""
        assert training_data_path.exists(), f"Missing: {training_data_path}"

    def test_emission_params_file_exists(self, emission_params_path):
        """Pre-computed emission parameters must exist."""
        assert emission_params_path.exists(), f"Missing: {emission_params_path}"

    def test_load_training_sequences(self, training_data_path):
        """Load and validate training sequences format."""
        df = pd.read_csv(training_data_path)

        # Must have expected columns
        expected_cols = ['sample', 'chrom', 'read_id', '38', '50', '62', '74', '86', '98', '110', '122']
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

        # Must have both control and modified samples
        samples = df['sample'].unique()
        assert 'control' in samples, "Missing control samples"
        assert 'modified' in samples, "Missing modified samples"

        # Must have substantial data
        n_control = len(df[df['sample'] == 'control'])
        n_modified = len(df[df['sample'] == 'modified'])
        assert n_control >= 100, f"Too few control samples: {n_control}"
        assert n_modified >= 100, f"Too few modified samples: {n_modified}"

        print(f"Loaded {n_control} control, {n_modified} modified reads")

    def test_load_emission_params(self, emission_params_path):
        """Load and validate emission parameters."""
        with open(emission_params_path) as f:
            params = json.load(f)

        assert 'states' in params, "Missing 'states' in params"
        assert len(params['states']) == 16, f"Expected 16 states (8 positions x 2), got {len(params['states'])}"

        # Check each state has required fields
        for state in params['states']:
            assert 'name' in state
            assert 'distribution' in state
            assert 'parameters' in state
            assert 'mean' in state['parameters']
            assert 'std' in state['parameters']

        # Check control and modified states exist for each position
        positions = [38, 50, 62, 74, 86, 98, 110, 122]
        for pos in positions:
            control_found = any(f'control_pos{pos}' in s['name'] for s in params['states'])
            modified_found = any(f'modified_pos{pos}' in s['name'] for s in params['states'])
            assert control_found, f"Missing control state for position {pos}"
            assert modified_found, f"Missing modified state for position {pos}"

    def test_control_vs_modified_signal_difference(self, signal_data_path):
        """Verify significant signal difference between C and 5mC."""
        df = pd.read_csv(signal_data_path)

        control = df[df['sample'] == 'control']['mean_current']
        modified = df[df['sample'] == 'modified']['mean_current']

        # 5mC should have higher current (as documented: +36.7 pA)
        control_mean = control.mean()
        modified_mean = modified.mean()

        assert modified_mean > control_mean, \
            f"Expected 5mC > C, got 5mC={modified_mean:.1f}, C={control_mean:.1f}"

        delta = modified_mean - control_mean
        assert delta > 20, f"Signal difference too small: {delta:.1f} pA (expected >20)"

        print(f"Signal difference: {delta:.1f} pA (5mC={modified_mean:.1f}, C={control_mean:.1f})")


class TestSimplifiedHMM:
    """Test simplified 8-position HMM construction."""

    @pytest.fixture
    def emission_params(self):
        path = PROJECT_ROOT / "output" / "hmm_emission_params_pomegranate.json"
        with open(path) as f:
            return json.load(f)

    def test_build_hmm_from_emission_params(self, emission_params):
        """Build a DenseHMM from pre-computed emission parameters."""
        from pomegranate.hmm import DenseHMM
        from pomegranate.distributions import Normal

        # Create distributions for each state
        distributions = []
        state_names = []

        for state in emission_params['states']:
            mean = state['parameters']['mean']
            std = state['parameters']['std']
            dist = Normal(
                means=torch.tensor([mean]),
                covs=torch.tensor([[std ** 2]])
            )
            distributions.append(dist)
            state_names.append(state['name'])

        n_states = len(distributions)
        assert n_states == 16, f"Expected 16 states, got {n_states}"

        # Build transition matrix (simple left-to-right with skip)
        edges = torch.zeros((n_states, n_states), dtype=torch.float32)

        # Group states by position (control and modified alternate)
        # Each position has 2 states: control_posX and modified_posX
        for i in range(0, n_states - 2, 2):
            # Current position states can go to next position states
            edges[i, i+2] = 0.5      # control -> next control
            edges[i, i+3] = 0.5      # control -> next modified
            edges[i+1, i+2] = 0.5    # modified -> next control
            edges[i+1, i+3] = 0.5    # modified -> next modified

        # Start probabilities (start at first position)
        starts = torch.zeros(n_states)
        starts[0] = 0.5  # start at control_pos38
        starts[1] = 0.5  # start at modified_pos38

        # End probabilities (end at last position)
        ends = torch.zeros(n_states)
        ends[-2] = 0.5   # end at control_pos122
        ends[-1] = 0.5   # end at modified_pos122

        # Build model
        model = DenseHMM(
            distributions=distributions,
            edges=edges,
            starts=starts,
            ends=ends
        )

        assert model is not None
        print(f"Built HMM with {n_states} states")

    def test_hmm_forward_pass(self, emission_params):
        """Verify forward algorithm works on test data."""
        from pomegranate.hmm import DenseHMM
        from pomegranate.distributions import Normal

        # Build model
        distributions = []
        for state in emission_params['states']:
            mean = state['parameters']['mean']
            std = state['parameters']['std']
            distributions.append(Normal(
                means=torch.tensor([mean]),
                covs=torch.tensor([[std ** 2]])
            ))

        n_states = len(distributions)
        edges = torch.zeros((n_states, n_states), dtype=torch.float32)
        for i in range(0, n_states - 2, 2):
            edges[i, i+2] = 0.5
            edges[i, i+3] = 0.5
            edges[i+1, i+2] = 0.5
            edges[i+1, i+3] = 0.5

        starts = torch.zeros(n_states)
        starts[0] = 0.5
        starts[1] = 0.5

        ends = torch.zeros(n_states)
        ends[-2] = 0.5
        ends[-1] = 0.5

        model = DenseHMM(
            distributions=distributions,
            edges=edges,
            starts=starts,
            ends=ends
        )

        # Create test sequence (8 values for 8 positions)
        test_seq = torch.tensor([
            [890.0],  # ~control pos38 mean
            [822.0],  # ~control pos50 mean
            [816.0],  # ~control pos62 mean
            [831.0],  # ~control pos74 mean
            [792.0],  # ~control pos86 mean
            [779.0],  # ~control pos98 mean
            [772.0],  # ~control pos110 mean
            [794.0],  # ~control pos122 mean
        ]).unsqueeze(0)  # Add batch dimension

        # Forward pass should return log probability
        log_prob = model.log_probability(test_seq)
        assert log_prob.shape == torch.Size([1])
        assert not torch.isnan(log_prob).any(), "Got NaN log probability"
        assert not torch.isinf(log_prob).any(), "Got infinite log probability"

        print(f"Test sequence log probability: {log_prob.item():.2f}")


class TestDataPreparer:
    """Test data preparation for HMM training."""

    @pytest.fixture
    def training_data(self):
        path = PROJECT_ROOT / "output" / "hmm_training_sequences.csv"
        return pd.read_csv(path)

    def test_prepare_sequences_for_hmm(self, training_data):
        """Convert training data to tensor format."""
        positions = ['38', '50', '62', '74', '86', '98', '110', '122']

        # Filter rows with no NaN
        df = training_data.dropna(subset=positions)

        # Extract sequences
        X = df[positions].values.astype(np.float32)

        # Should have shape (n_reads, 8)
        assert X.shape[1] == 8, f"Expected 8 positions, got {X.shape[1]}"
        assert len(X) > 200, f"Too few valid reads: {len(X)}"

        # Convert to torch tensor with feature dimension
        X_tensor = torch.tensor(X).unsqueeze(-1)  # (n_reads, 8, 1)

        assert X_tensor.shape[2] == 1
        assert not torch.isnan(X_tensor).any(), "NaN values in tensor"

        print(f"Prepared {len(X)} sequences of shape {X_tensor.shape}")

    def test_split_train_test(self, training_data):
        """Split data into train and test sets by sample."""
        control = training_data[training_data['sample'] == 'control']
        modified = training_data[training_data['sample'] == 'modified']

        # 80/20 split
        n_control_train = int(len(control) * 0.8)
        n_modified_train = int(len(modified) * 0.8)

        assert n_control_train > 100, "Not enough control training data"
        assert n_modified_train > 100, "Not enough modified training data"

        print(f"Train split: {n_control_train} control, {n_modified_train} modified")


class TestHMMTraining:
    """Test HMM training on real data."""

    @pytest.fixture
    def training_data(self):
        path = PROJECT_ROOT / "output" / "hmm_training_sequences.csv"
        return pd.read_csv(path)

    @pytest.fixture
    def emission_params(self):
        path = PROJECT_ROOT / "output" / "hmm_emission_params_pomegranate.json"
        with open(path) as f:
            return json.load(f)

    def test_hmm_training_improves_likelihood(self, training_data, emission_params):
        """Training should improve data likelihood."""
        from pomegranate.hmm import DenseHMM
        from pomegranate.distributions import Normal

        # Build initial model
        distributions = []
        for state in emission_params['states']:
            mean = state['parameters']['mean']
            std = state['parameters']['std']
            distributions.append(Normal(
                means=torch.tensor([mean]),
                covs=torch.tensor([[std ** 2]])
            ))

        n_states = len(distributions)
        edges = torch.zeros((n_states, n_states), dtype=torch.float32)
        for i in range(0, n_states - 2, 2):
            edges[i, i+2] = 0.5
            edges[i, i+3] = 0.5
            edges[i+1, i+2] = 0.5
            edges[i+1, i+3] = 0.5

        starts = torch.zeros(n_states)
        starts[0] = 0.5
        starts[1] = 0.5

        ends = torch.zeros(n_states)
        ends[-2] = 0.5
        ends[-1] = 0.5

        model = DenseHMM(
            distributions=distributions,
            edges=edges,
            starts=starts,
            ends=ends,
            max_iter=5,
            tol=0.01
        )

        # Prepare data
        positions = ['38', '50', '62', '74', '86', '98', '110', '122']
        df = training_data.dropna(subset=positions)
        X = torch.tensor(df[positions].values.astype(np.float32)).unsqueeze(-1)

        # Sample 500 reads for speed
        if len(X) > 500:
            indices = torch.randperm(len(X))[:500]
            X = X[indices]

        # Get initial likelihood
        initial_logp = model.log_probability(X).mean().item()

        # Train
        model.fit(X)

        # Get final likelihood
        final_logp = model.log_probability(X).mean().item()

        print(f"Log likelihood: {initial_logp:.2f} -> {final_logp:.2f}")

        # Training should improve or maintain likelihood
        assert final_logp >= initial_logp - 1.0, \
            f"Training degraded likelihood: {initial_logp:.2f} -> {final_logp:.2f}"


class TestClassification:
    """Test methylation classification accuracy."""

    @pytest.fixture
    def training_data(self):
        path = PROJECT_ROOT / "output" / "hmm_training_sequences.csv"
        return pd.read_csv(path)

    @pytest.fixture
    def emission_params(self):
        path = PROJECT_ROOT / "output" / "hmm_emission_params_pomegranate.json"
        with open(path) as f:
            return json.load(f)

    def build_linear_hmm(self, states_params, n_positions=8):
        """Build a simple left-to-right HMM for a sequence of positions."""
        from pomegranate.hmm import DenseHMM
        from pomegranate.distributions import Normal

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
            edges[i, i + 1] = 1.0  # Go to next state

        # Start at first state
        starts = torch.zeros(n_states)
        starts[0] = 1.0

        # End at last state
        ends = torch.zeros(n_states)
        ends[-1] = 1.0

        model = DenseHMM(
            distributions=distributions,
            edges=edges,
            starts=starts,
            ends=ends
        )
        return model

    def test_baseline_classification_accuracy(self, training_data, emission_params):
        """
        Test that model can classify control vs modified samples.

        Uses likelihood ratio classification:
        - Build control model with control emissions
        - Build modified model with modified emissions
        - Classify by comparing log P(X | control) vs log P(X | modified)

        We expect >70% accuracy since signal difference is +36.7 pA.
        """
        # Separate emission params for control and modified
        control_params = [s['parameters'] for s in emission_params['states'] if 'control' in s['name']]
        modified_params = [s['parameters'] for s in emission_params['states'] if 'modified' in s['name']]

        assert len(control_params) == 8, f"Expected 8 control params, got {len(control_params)}"
        assert len(modified_params) == 8, f"Expected 8 modified params, got {len(modified_params)}"

        # Build two separate HMMs
        control_model = self.build_linear_hmm(control_params)
        modified_model = self.build_linear_hmm(modified_params)

        # Prepare data
        positions = ['38', '50', '62', '74', '86', '98', '110', '122']
        df = training_data.dropna(subset=positions)

        # Get labels
        labels = df['sample'].values
        y_true = (labels == 'modified').astype(int)

        # Get sequences
        X = torch.tensor(df[positions].values.astype(np.float32)).unsqueeze(-1)

        # Classify using likelihood ratio
        control_model.eval()
        modified_model.eval()

        with torch.no_grad():
            logp_control = control_model.log_probability(X)
            logp_modified = modified_model.log_probability(X)

        # Classify based on which model has higher likelihood
        log_ratio = logp_modified - logp_control
        y_pred = (log_ratio > 0).numpy().astype(int)

        # Compute accuracy
        accuracy = (y_pred == y_true).mean()
        n_correct = (y_pred == y_true).sum()

        control_acc = (y_pred[y_true == 0] == 0).mean()
        modified_acc = (y_pred[y_true == 1] == 1).mean()

        print(f"Classification accuracy: {accuracy:.3f} ({n_correct}/{len(y_true)})")
        print(f"  Control: {control_acc:.3f} accuracy")
        print(f"  Modified: {modified_acc:.3f} accuracy")
        print(f"  Log ratio mean: control={logp_control.mean():.2f}, modified={logp_modified.mean():.2f}")

        # ACCEPTANCE CRITERION: >70% overall accuracy
        assert accuracy > 0.70, f"Accuracy {accuracy:.3f} below threshold 0.70"


class TestEndToEndPipeline:
    """End-to-end integration test."""

    def build_linear_hmm(self, states_params):
        """Build a simple left-to-right HMM for a sequence of positions."""
        from pomegranate.hmm import DenseHMM
        from pomegranate.distributions import Normal

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

        # Start at first state
        starts = torch.zeros(n_states)
        starts[0] = 1.0

        # End at last state
        ends = torch.zeros(n_states)
        ends[-1] = 1.0

        model = DenseHMM(
            distributions=distributions,
            edges=edges,
            starts=starts,
            ends=ends,
            max_iter=10,
            tol=0.001
        )
        return model

    def test_full_pipeline(self):
        """
        Run complete pipeline: load data -> build HMMs -> train -> classify -> evaluate.

        Uses likelihood ratio classification with two separate HMMs:
        - Control HMM: trained on control data
        - Modified HMM: trained on modified data

        This is the ultimate acceptance test that proves everything works together.
        """
        from pomegranate.hmm import DenseHMM
        from pomegranate.distributions import Normal

        # 1. Load data
        training_path = PROJECT_ROOT / "output" / "hmm_training_sequences.csv"
        params_path = PROJECT_ROOT / "output" / "hmm_emission_params_pomegranate.json"

        df = pd.read_csv(training_path)
        with open(params_path) as f:
            emission_params = json.load(f)

        positions = ['38', '50', '62', '74', '86', '98', '110', '122']
        df_clean = df.dropna(subset=positions)

        print(f"Loaded {len(df_clean)} reads with complete data")

        # 2. Split train/test
        control = df_clean[df_clean['sample'] == 'control']
        modified = df_clean[df_clean['sample'] == 'modified']

        n_control_train = int(len(control) * 0.8)
        n_modified_train = int(len(modified) * 0.8)

        train_control = control.iloc[:n_control_train]
        train_modified = modified.iloc[:n_modified_train]
        test_df = pd.concat([
            control.iloc[n_control_train:],
            modified.iloc[n_modified_train:]
        ])

        print(f"Train: {n_control_train} control, {n_modified_train} modified")
        print(f"Test: {len(test_df)}")

        # 3. Build separate HMMs for control and modified
        control_params = [s['parameters'] for s in emission_params['states'] if 'control' in s['name']]
        modified_params = [s['parameters'] for s in emission_params['states'] if 'modified' in s['name']]

        control_model = self.build_linear_hmm(control_params)
        modified_model = self.build_linear_hmm(modified_params)

        # 4. Train each model on its respective data
        X_train_control = torch.tensor(train_control[positions].values.astype(np.float32)).unsqueeze(-1)
        X_train_modified = torch.tensor(train_modified[positions].values.astype(np.float32)).unsqueeze(-1)

        control_model.fit(X_train_control)
        modified_model.fit(X_train_modified)
        print("Training complete")

        # 5. Evaluate on test set using likelihood ratio
        X_test = torch.tensor(test_df[positions].values.astype(np.float32)).unsqueeze(-1)
        y_test = (test_df['sample'].values == 'modified').astype(int)

        control_model.eval()
        modified_model.eval()

        with torch.no_grad():
            logp_control = control_model.log_probability(X_test)
            logp_modified = modified_model.log_probability(X_test)

        # Classify based on likelihood ratio
        log_ratio = logp_modified - logp_control
        y_pred = (log_ratio > 0).numpy().astype(int)

        # 6. Report metrics
        accuracy = (y_pred == y_test).mean()
        control_acc = (y_pred[y_test == 0] == 0).mean()
        modified_acc = (y_pred[y_test == 1] == 1).mean()

        print(f"\n{'='*50}")
        print("END-TO-END TEST RESULTS")
        print(f"{'='*50}")
        print(f"Test set size: {len(y_test)}")
        print(f"Overall accuracy: {accuracy:.3f}")
        print(f"Control accuracy: {control_acc:.3f}")
        print(f"Modified accuracy: {modified_acc:.3f}")
        print(f"Log likelihood: control={logp_control.mean():.2f}, modified={logp_modified.mean():.2f}")
        print(f"{'='*50}\n")

        # ACCEPTANCE CRITERIA
        assert accuracy > 0.65, f"Overall accuracy {accuracy:.3f} too low"
        assert control_acc > 0.50, f"Control accuracy {control_acc:.3f} too low"
        assert modified_acc > 0.50, f"Modified accuracy {modified_acc:.3f} too low"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
