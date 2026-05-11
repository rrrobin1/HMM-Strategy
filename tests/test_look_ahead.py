"""
test_look_ahead.py — Prove no look-ahead bias in HMM and feature pipeline.

MANDATORY TEST (Phase 2 spec):
  test_no_look_ahead_bias — the canonical proof:
    1. Train HMM on first 600 feature-bars.
    2. Run forward filter up to bar 600 → record alpha at bar 600.
    3. Run forward filter up to bar 601 (one extra bar).
    4. ASSERT alpha at bar 600 is numerically identical in both runs.
  Proves alpha_t depends only on obs_1:t, never on obs_{t+1:T}.

Note: make_bars(N) → approximately (N - 300) feature rows after rolling
z-score windows are filled. make_bars(1000) → ~700 feature rows.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from core.hmm_engine import HMMConfig, HMMEngine
from data.feature_engineering import FeatureEngineer


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def make_bars(n: int = 1000, seed: int = 7) -> pd.DataFrame:
    """Synthetic OHLCV with alternating vol regimes every 60 bars."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n)
    cycle = (np.arange(n) // 60) % 2
    vol = np.where(cycle == 0, 0.007, 0.022)
    drift = np.where(cycle == 0, 0.0004, -0.0003)
    log_ret = rng.normal(drift, vol)
    close = 100.0 * np.exp(np.cumsum(log_ret))
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.006,
            "low": close * 0.994,
            "close": close,
            "volume": rng.integers(500_000, 3_000_000, size=n),
        },
        index=dates,
    )


def make_config() -> HMMConfig:
    return HMMConfig(
        n_candidates=[2, 3],
        n_init=3,
        min_train_bars=200,
        stability_bars=2,
        flicker_window=10,
        flicker_threshold=3,
        min_confidence=0.55,
    )


# ------------------------------------------------------------------ #
# MANDATORY: one-bar-ahead test                                        #
# ------------------------------------------------------------------ #

def test_no_look_ahead_bias():
    """
    THE canonical look-ahead bias test (Phase 2 spec).

    Verifies that the filtered alpha at bar T is identical whether
    computed from obs_1:T or from obs_1:T+1. If this fails, the
    forward algorithm is incorrectly using future data.
    """
    bars = make_bars(1000)
    full_features = FeatureEngineer.build_hmm_features(bars)
    assert len(full_features) >= 650, (
        f"Need >= 650 feature rows, got {len(full_features)}. "
        "Increase make_bars n."
    )

    engine = HMMEngine(make_config())
    engine.fit(full_features.iloc[:600])

    # Target bar: row 600 in the feature array.
    target = 600

    # Run forward filter on obs_1 through obs_{target+1}
    alpha_at_target = engine._forward_filter(full_features.iloc[: target + 1].values)
    label_at = int(np.argmax(alpha_at_target[-1]))

    # Run forward filter on obs_1 through obs_{target+2} (one extra bar)
    alpha_plus_one = engine._forward_filter(full_features.iloc[: target + 2].values)
    label_after_extra = int(np.argmax(alpha_plus_one[target]))

    assert label_at == label_after_extra, (
        f"Look-ahead bias detected! "
        f"Alpha at bar {target} changed label from {label_at} to "
        f"{label_after_extra} when bar {target + 1} was added."
    )

    np.testing.assert_allclose(
        alpha_at_target[-1],
        alpha_plus_one[target],
        rtol=1e-10,
        err_msg=f"Alpha vector at bar {target} changed after adding bar {target + 1}.",
    )


# ------------------------------------------------------------------ #
# Parametric HMM look-ahead tests                                      #
# ------------------------------------------------------------------ #

class TestHMMNoLookAhead:
    """
    Verify forward filter alpha[t] on truncated features matches alpha[t]
    in the full forward pass. Uses make_bars(1000) → ~700 feature rows;
    training on first 600, testing at bars 600-650.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        bars = make_bars(1000)
        features = FeatureEngineer.build_hmm_features(bars)
        engine = HMMEngine(make_config())
        engine.fit(features.iloc[:600])
        self.engine = engine
        self.full_features = features
        self.full_alpha = engine._forward_filter(features.values)
        self.n_features = len(features)

    @pytest.mark.parametrize("offset", [0, 10, 20, 30, 40])
    def test_alpha_unchanged_at_bar(self, offset: int):
        """Alpha at bar (600+offset) must not change when future bars are present."""
        bar_idx = 600 + offset
        if bar_idx >= self.n_features:
            pytest.skip(f"bar_idx={bar_idx} >= n_features={self.n_features}")

        alpha_truncated = self.engine._forward_filter(
            self.full_features.iloc[: bar_idx + 1].values
        )
        np.testing.assert_allclose(
            alpha_truncated[-1],
            self.full_alpha[bar_idx],
            rtol=1e-10,
            err_msg=f"Look-ahead detected at bar {bar_idx}.",
        )

    @pytest.mark.parametrize("offset", [0, 20, 40])
    def test_predict_matches_sequence_at_bar(self, offset: int):
        """predict() on truncated features must match predict_sequence()[bar_idx]."""
        bar_idx = 600 + offset
        if bar_idx >= self.n_features:
            pytest.skip(f"bar_idx={bar_idx} >= n_features={self.n_features}")

        seq_states = self.engine.predict_sequence(self.full_features)
        pred_state = self.engine.predict(self.full_features.iloc[: bar_idx + 1])
        assert pred_state.state_id == seq_states[bar_idx].state_id, (
            f"Mismatch at bar {bar_idx}: "
            f"predict={pred_state.label}, sequence={seq_states[bar_idx].label}"
        )


# ------------------------------------------------------------------ #
# Feature look-ahead tests                                             #
# ------------------------------------------------------------------ #

class TestFeatureNoLookAhead:
    """
    Indicator computed on bars[0:N] must equal indicator at row N in the
    full-series computation. Uses make_bars(600) → ~300 feature rows.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.bars = make_bars(600)
        self.enriched_full = FeatureEngineer.enrich_bars(self.bars.copy())
        self.features_full = FeatureEngineer.build_hmm_features(self.bars.copy())
        self.n_features = len(self.features_full)

    @pytest.mark.parametrize("col", ["ema_50", "atr", "rsi", "adx"])
    @pytest.mark.parametrize("bar_idx", [200, 300, 400, 500])
    def test_strategy_indicator_no_look_ahead(self, col: str, bar_idx: int):
        if bar_idx >= len(self.bars):
            pytest.skip(f"bar_idx={bar_idx} >= n_bars={len(self.bars)}")
        truncated = self.bars.iloc[: bar_idx + 1].copy()
        enriched_t = FeatureEngineer.enrich_bars(truncated)
        val_t = enriched_t[col].iloc[-1]
        val_f = self.enriched_full[col].iloc[bar_idx]
        assert abs(val_t - val_f) < 1e-8, (
            f"Look-ahead in {col} at bar {bar_idx}: "
            f"truncated={val_t:.8f}, full={val_f:.8f}"
        )

    @pytest.mark.parametrize("col", ["log_ret_1", "realized_vol_20", "norm_atr"])
    def test_hmm_feature_no_look_ahead(self, col: str):
        """HMM features at a bar must be identical with or without future bars."""
        # Use a feature row well inside the available range.
        feat_idx = min(50, self.n_features - 1)
        bar_date = self.features_full.index[feat_idx]
        raw_idx = self.bars.index.get_loc(bar_date)
        truncated_bars = self.bars.iloc[: raw_idx + 1]
        features_t = FeatureEngineer.build_hmm_features(truncated_bars)

        val_t = features_t[col].iloc[-1]
        val_f = self.features_full[col].iloc[feat_idx]
        assert abs(val_t - val_f) < 1e-8, (
            f"HMM feature look-ahead in {col}: "
            f"truncated={val_t:.8f}, full={val_f:.8f}"
        )
