"""
test_hmm.py — Unit tests for HMMEngine (Phase 2 implementation).

Covers:
  - fit() succeeds on sufficient data and raises on insufficient data
  - Selected n_states is within n_candidates
  - State labels are sorted by return (lowest return → index 0)
  - predict() returns a valid RegimeState with correct field types
  - state_probabilities sums to 1
  - predict() before fit raises RuntimeError
  - predict_sequence() length matches feature rows
  - is_fitted() reflects state correctly
  - save() / load() round-trip preserves model

Note: make_bars(1000) → ~700 feature rows (rolling z-score drops ~300 initial bars).
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.hmm_engine import HMMConfig, HMMEngine, RegimeState
from data.feature_engineering import FeatureEngineer


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #

def make_bars(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLCV with two alternating vol regimes."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n)
    cycle = (np.arange(n) // 60) % 2
    vol = np.where(cycle == 0, 0.007, 0.025)
    drift = np.where(cycle == 0, 0.0005, -0.0002)
    log_ret = rng.normal(drift, vol)
    close = 100.0 * np.exp(np.cumsum(log_ret))
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.006,
            "low": close * 0.994,
            "close": close,
            "volume": rng.integers(500_000, 5_000_000, n),
        },
        index=dates,
    )


@pytest.fixture(scope="module")
def bars_large():
    return make_bars(1000)


@pytest.fixture(scope="module")
def features_large(bars_large):
    return FeatureEngineer.build_hmm_features(bars_large)


@pytest.fixture
def config() -> HMMConfig:
    return HMMConfig(
        n_candidates=[2, 3],
        n_init=3,
        min_train_bars=200,
        stability_bars=2,
        flicker_window=10,
        flicker_threshold=3,
        min_confidence=0.55,
    )


@pytest.fixture(scope="module")
def fitted_engine(features_large) -> HMMEngine:
    cfg = HMMConfig(n_candidates=[2, 3], n_init=3, min_train_bars=200)
    engine = HMMEngine(cfg)
    engine.fit(features_large.iloc[:500])
    return engine


# ------------------------------------------------------------------ #
# Fit                                                                  #
# ------------------------------------------------------------------ #

class TestHMMFit:
    def test_fit_sets_is_fitted(self, fitted_engine):
        assert fitted_engine.is_fitted()

    def test_fresh_engine_not_fitted(self, config):
        assert not HMMEngine(config).is_fitted()

    def test_fit_raises_on_insufficient_data(self, config, features_large):
        engine = HMMEngine(config)
        with pytest.raises(ValueError, match="at least"):
            engine.fit(features_large.iloc[:50])

    def test_n_states_within_candidates(self, fitted_engine, config):
        assert fitted_engine._n_states in [2, 3]

    def test_state_labels_match_n_states(self, fitted_engine):
        assert len(fitted_engine._state_labels) == fitted_engine._n_states

    def test_regime_infos_populated(self, fitted_engine):
        infos = fitted_engine.get_regime_infos()
        assert len(infos) == fitted_engine._n_states
        for rank, info in infos.items():
            assert info.vol_environment in ("low", "mid", "high")
            assert info.recommended_strategy_type in (
                "low_vol_bull", "mid_vol_cautious", "high_vol_defensive"
            )


# ------------------------------------------------------------------ #
# Predict                                                              #
# ------------------------------------------------------------------ #

class TestPredict:
    def test_returns_regime_state(self, fitted_engine, features_large):
        state = fitted_engine.predict(features_large.iloc[:510])
        assert isinstance(state, RegimeState)

    def test_label_is_string(self, fitted_engine, features_large):
        state = fitted_engine.predict(features_large.iloc[:510])
        assert isinstance(state.label, str)
        assert state.label in fitted_engine._state_labels

    def test_probability_in_unit_interval(self, fitted_engine, features_large):
        state = fitted_engine.predict(features_large.iloc[:510])
        assert 0.0 <= state.probability <= 1.0

    def test_state_probabilities_sum_to_one(self, fitted_engine, features_large):
        state = fitted_engine.predict(features_large.iloc[:510])
        assert abs(state.state_probabilities.sum() - 1.0) < 1e-6

    def test_vol_environment_valid(self, fitted_engine, features_large):
        state = fitted_engine.predict(features_large.iloc[:510])
        assert state.vol_environment in ("low", "mid", "high")

    def test_predict_before_fit_raises(self, config, features_large):
        engine = HMMEngine(config)
        with pytest.raises(RuntimeError):
            engine.predict(features_large.iloc[:300])


# ------------------------------------------------------------------ #
# Predict sequence                                                     #
# ------------------------------------------------------------------ #

class TestPredictSequence:
    def test_length_matches_features(self, fitted_engine, features_large):
        window = features_large.iloc[300:550]
        states = fitted_engine.predict_sequence(window)
        assert len(states) == len(window)

    def test_all_are_regime_states(self, fitted_engine, features_large):
        states = fitted_engine.predict_sequence(features_large.iloc[300:400])
        assert all(isinstance(s, RegimeState) for s in states)

    def test_timestamps_aligned(self, fitted_engine, features_large):
        window = features_large.iloc[300:350]
        states = fitted_engine.predict_sequence(window)
        for i, state in enumerate(states):
            assert state.timestamp == window.index[i]


# ------------------------------------------------------------------ #
# Return-sorted labels                                                 #
# ------------------------------------------------------------------ #

class TestReturnSortedLabels:
    def test_label_at_index_0_is_lowest_return(self, fitted_engine, features_large):
        """
        The raw state mapped to return_rank=0 must have the lowest mean
        feature-0 value (log_ret_1) in the training data.
        """
        X = features_large.iloc[:500].values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assignments = fitted_engine._model.predict(X)

        means = np.array([
            X[assignments == k, 0].mean() if (assignments == k).any() else 0.0
            for k in range(fitted_engine._n_states)
        ])
        lowest_raw = int(np.argmin(means))
        lowest_rank = int(fitted_engine._raw_to_return_rank[lowest_raw])
        assert lowest_rank == 0, (
            f"Lowest-return raw state {lowest_raw} should map to rank 0, "
            f"got rank {lowest_rank}. Means: {means}"
        )


# ------------------------------------------------------------------ #
# Stability and flicker                                                #
# ------------------------------------------------------------------ #

class TestStabilityAndFlicker:
    def test_get_regime_stability_non_negative(self, fitted_engine, features_large):
        fitted_engine.predict_sequence(features_large.iloc[300:400])
        assert fitted_engine.get_regime_stability() >= 0

    def test_flicker_rate_non_negative(self, fitted_engine, features_large):
        fitted_engine.predict_sequence(features_large.iloc[300:400])
        assert fitted_engine.get_regime_flicker_rate() >= 0.0

    def test_is_flickering_bool(self, fitted_engine, features_large):
        fitted_engine.predict_sequence(features_large.iloc[300:400])
        assert isinstance(fitted_engine.is_flickering(), bool)


# ------------------------------------------------------------------ #
# Persistence                                                          #
# ------------------------------------------------------------------ #

class TestPersistence:
    def test_save_load_roundtrip(self, fitted_engine, features_large):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_model.pkl"
            fitted_engine.save(path)
            loaded = HMMEngine.load(path)

        assert loaded._n_states == fitted_engine._n_states
        assert loaded._state_labels == fitted_engine._state_labels

        state_orig = fitted_engine.predict(features_large.iloc[:520])
        state_load = loaded.predict(features_large.iloc[:520])
        assert state_orig.label == state_load.label
