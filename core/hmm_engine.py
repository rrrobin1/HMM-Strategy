"""
hmm_engine.py — Hidden Markov Model regime detection engine.

DESIGN: The HMM is a VOLATILITY CLASSIFIER. It identifies whether the market
is in a calm, moderate, or turbulent environment. It does NOT predict price
direction. The strategy layer uses the classification to set allocation.

KEY INVARIANT — NO LOOK-AHEAD BIAS:
  model.predict() (Viterbi) is never used for live/backtest decisions because
  it revises past states using future observations. Instead, the forward
  algorithm is used: alpha_t = P(state_t | obs_1:t). This is strictly causal.
  See predict_regime_filtered() and _forward_filter().

Model selection: BIC across n_candidates state counts, n_init restarts each.
State labels: sorted by expected return (lowest → CRASH/BEAR, highest → BULL/EUPHORIA).
Vol environment: independently sorted by expected volatility → "low" | "mid" | "high".
"""

from __future__ import annotations

import logging
import pickle
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

logger = logging.getLogger(__name__)

# Human-readable label sets indexed by n_states.
_LABELS: dict[int, list[str]] = {
    3: ["BEAR", "NEUTRAL", "BULL"],
    4: ["CRASH", "BEAR", "BULL", "EUPHORIA"],
    5: ["CRASH", "BEAR", "NEUTRAL", "BULL", "EUPHORIA"],
    6: ["CRASH", "STRONG_BEAR", "WEAK_BEAR", "WEAK_BULL", "STRONG_BULL", "EUPHORIA"],
    7: ["CRASH", "STRONG_BEAR", "WEAK_BEAR", "NEUTRAL", "WEAK_BULL", "STRONG_BULL", "EUPHORIA"],
}


# ------------------------------------------------------------------ #
# Configuration                                                        #
# ------------------------------------------------------------------ #

@dataclass
class HMMConfig:
    """Parameters loaded from settings.yaml[hmm]."""

    n_candidates: list[int] = field(default_factory=lambda: [3, 4, 5, 6, 7])
    n_init: int = 10
    covariance_type: str = "full"
    min_train_bars: int = 504           # 2 years of daily data
    stability_bars: int = 3             # bars before a regime change is confirmed
    flicker_window: int = 20            # bars in the flicker-rate window
    flicker_threshold: int = 4          # max changes per flicker_window
    min_confidence: float = 0.55        # min posterior to act on a regime


# ------------------------------------------------------------------ #
# Output dataclasses                                                   #
# ------------------------------------------------------------------ #

@dataclass
class RegimeInfo:
    """Static metadata about one fitted regime state."""

    regime_id: int                      # return-sorted rank (0 = lowest return)
    regime_name: str                    # e.g. "BULL", "CRASH"
    expected_return: float              # annualized mean log return in training data
    expected_volatility: float          # annualized realized vol in training data
    vol_environment: str                # "low" | "mid" | "high"
    recommended_strategy_type: str      # "low_vol_bull" | "mid_vol_cautious" | "high_vol_defensive"
    max_leverage_allowed: float = 1.0
    max_position_size_pct: float = 0.15
    min_confidence_to_act: float = 0.55


@dataclass
class RegimeState:
    """Output produced by HMMEngine for a single bar (forward algorithm)."""

    label: str                          # "BULL", "BEAR", "CRASH", etc.
    state_id: int                       # return-sorted rank (0 = lowest return)
    probability: float                  # P(most-likely state | obs_1:t)
    state_probabilities: np.ndarray     # full filtered posterior, shape (n_states,)
    timestamp: Optional[pd.Timestamp]
    is_confirmed: bool                  # True when stable for >= stability_bars bars
    consecutive_bars: int               # bars since last regime change was confirmed
    vol_environment: str                # "low" | "mid" | "high"
    n_states: int = 0
    regime_info: Optional[RegimeInfo] = None

    # Backward-compatible aliases used by earlier test stubs.
    @property
    def confidence(self) -> float:
        return self.probability

    @property
    def posteriors(self) -> np.ndarray:
        return self.state_probabilities

    @property
    def is_stable(self) -> bool:
        return self.is_confirmed


# ------------------------------------------------------------------ #
# HMM Engine                                                           #
# ------------------------------------------------------------------ #

class HMMEngine:
    """
    Fits a Gaussian HMM on log-return features and emits causal regime states.

    Workflow
    --------
    1.  engine.fit(features)          — train, select n_states, build metadata
    2.  engine.predict(features)      — current-bar regime via forward algorithm
    3.  engine.predict_sequence(features) — all-bar sequence (backtester)

    Thread safety: not thread-safe; use one instance per process.
    """

    def __init__(self, config: HMMConfig) -> None:
        self.config = config
        self._model = None              # best-fit hmmlearn.GaussianHMM
        self._n_states: int = 0
        self._state_labels: list[str] = []

        # Mapping arrays: convert between raw HMM state index and sorted ranks.
        self._return_sort_to_raw: np.ndarray = np.array([])   # return_rank → raw_state
        self._raw_to_return_rank: np.ndarray = np.array([])   # raw_state   → return_rank
        self._vol_sort_to_raw: np.ndarray = np.array([])      # vol_rank    → raw_state
        self._raw_to_vol_rank: np.ndarray = np.array([])      # raw_state   → vol_rank

        self._regime_infos: dict[int, RegimeInfo] = {}        # return_rank → RegimeInfo

        # Stability-filter state (mutated by _update_stability).
        self._confirmed_regime: int = -1
        self._consecutive_bars: int = 0
        self._pending_regime: int = -1
        self._pending_count: int = 0
        self._regime_history: list[int] = []    # raw states per bar

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, features: pd.DataFrame) -> None:
        """
        Fit the HMM using BIC model selection over n_candidates state counts.

        Parameters
        ----------
        features:
            DataFrame produced by FeatureEngineer.build_hmm_features().
            Must have at least config.min_train_bars rows.

        Raises
        ------
        ValueError
            If fewer than min_train_bars rows are available.
        RuntimeError
            If no candidate converges.
        """
        if len(features) < self.config.min_train_bars:
            raise ValueError(
                f"Need at least {self.config.min_train_bars} bars to fit HMM, "
                f"got {len(features)}."
            )

        X = features.values.astype(np.float64)
        n_features = X.shape[1]

        best_model = None
        best_bic = np.inf
        bic_scores: dict[int, float] = {}

        for n in self.config.n_candidates:
            model, ll = self._fit_candidate(X, n)
            if model is None:
                logger.warning("n_states=%d: all initializations failed, skipping.", n)
                continue
            bic = self._compute_bic(model, X, n_features)
            bic_scores[n] = bic
            logger.info("n_states=%d  ll=%.2f  BIC=%.2f", n, ll, bic)
            if bic < best_bic:
                best_bic = bic
                best_model = model

        if best_model is None:
            raise RuntimeError("HMM fitting failed for all n_candidates.")

        self._model = best_model
        self._n_states = best_model.n_components
        logger.info(
            "Selected n_states=%d (BIC=%.2f). All BICs: %s",
            self._n_states, best_bic,
            {k: f"{v:.1f}" for k, v in bic_scores.items()},
        )

        self._build_sort_mappings(X)
        self._build_regime_infos(X)
        self._reset_stability()

    def _fit_candidate(
        self, X: np.ndarray, n: int
    ) -> tuple[object, float]:
        """
        Fit n_init models with n states; return (best_model, best_ll).
        Returns (None, -inf) if every initialization fails.
        """
        from hmmlearn.hmm import GaussianHMM

        best_model = None
        best_ll = -np.inf

        for seed in range(self.config.n_init):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    model = GaussianHMM(
                        n_components=n,
                        covariance_type=self.config.covariance_type,
                        n_iter=100,
                        tol=1e-4,
                        random_state=seed,
                        verbose=False,
                    )
                    model.fit(X)
                    ll = model.score(X)
                    if ll > best_ll:
                        best_ll = ll
                        best_model = model
                except Exception as exc:
                    logger.debug("n_states=%d seed=%d failed: %s", n, seed, exc)

        return best_model, best_ll

    @staticmethod
    def _compute_bic(model, X: np.ndarray, n_features: int) -> float:
        """BIC = -2 * log_likelihood + n_params * log(n_samples)."""
        n = model.n_components
        n_params = (
            (n - 1)                                         # start probs
            + n * (n - 1)                                   # transition matrix
            + n * n_features                                # means
            + n * n_features * (n_features + 1) // 2       # full covariance
        )
        total_ll = model.score(X)
        return -2.0 * total_ll + n_params * np.log(len(X))

    def _build_sort_mappings(self, X: np.ndarray) -> None:
        """
        Compute two sort orders:
          1. By expected return (ascending)  → human-readable labels
          2. By expected volatility (ascending) → vol_environment
        """
        # Use Viterbi on training data to assign bars to states.
        # Viterbi is acceptable here because this is metadata computation
        # after fitting, not a trading decision.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_assignments = self._model.predict(X)

        # Feature index 0 = log_ret_1 (standardized), good proxy for return.
        # Feature index 3 = realized_vol_20, good proxy for volatility.
        return_means = np.array([
            X[train_assignments == k, 0].mean() if (train_assignments == k).any() else 0.0
            for k in range(self._n_states)
        ])
        vol_means = np.array([
            X[train_assignments == k, 3].mean() if (train_assignments == k).any() else 0.0
            for k in range(self._n_states)
        ])

        self._return_sort_to_raw = np.argsort(return_means)        # rank → raw
        self._raw_to_return_rank = np.argsort(self._return_sort_to_raw)  # raw → rank
        self._vol_sort_to_raw = np.argsort(vol_means)              # vol_rank → raw
        self._raw_to_vol_rank = np.argsort(self._vol_sort_to_raw)  # raw → vol_rank

        self._state_labels = _LABELS.get(
            self._n_states,
            [str(i) for i in range(self._n_states)],
        )

    def _build_regime_infos(self, X: np.ndarray) -> None:
        """
        Populate _regime_infos: one RegimeInfo per return-sorted rank.
        Expected return/vol are computed from raw feature values in training data.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assignments = self._model.predict(X)

        self._regime_infos = {}
        n = self._n_states

        for raw_state in range(n):
            return_rank = int(self._raw_to_return_rank[raw_state])
            vol_rank = int(self._raw_to_vol_rank[raw_state])

            mask = assignments == raw_state
            # Feature 0 = standardized log_ret_1 — not in real units, but relative
            exp_ret = float(X[mask, 0].mean()) if mask.any() else 0.0
            # Feature 3 = standardized realized_vol_20
            exp_vol = float(X[mask, 3].mean()) if mask.any() else 0.0

            vol_env = self._vol_rank_to_env(vol_rank, n)
            strategy = self._vol_env_to_strategy(vol_env)
            max_lev = {
                "low": 1.25,
                "mid": 1.0,
                "high": 0.6,
            }[vol_env]

            self._regime_infos[return_rank] = RegimeInfo(
                regime_id=return_rank,
                regime_name=self._state_labels[return_rank],
                expected_return=exp_ret,
                expected_volatility=exp_vol,
                vol_environment=vol_env,
                recommended_strategy_type=strategy,
                max_leverage_allowed=max_lev,
                max_position_size_pct=0.15,
                min_confidence_to_act=self.config.min_confidence,
            )

    @staticmethod
    def _vol_rank_to_env(vol_rank: int, n_states: int) -> str:
        frac = vol_rank / max(n_states - 1, 1)
        if frac <= 0.33:
            return "low"
        if frac <= 0.67:
            return "mid"
        return "high"

    @staticmethod
    def _vol_env_to_strategy(vol_env: str) -> str:
        return {
            "low": "low_vol_bull",
            "mid": "mid_vol_cautious",
            "high": "high_vol_defensive",
        }[vol_env]

    # ------------------------------------------------------------------
    # Forward algorithm (causal regime prediction)
    # ------------------------------------------------------------------

    def _log_emission_probs(self, X: np.ndarray) -> np.ndarray:
        """
        Compute log P(obs_t | state_k) for all t and k.

        Returns shape (T, n_states). Uses multivariate Gaussian emission
        with the fitted means and covariances.
        """
        T = len(X)
        n = self._n_states
        log_probs = np.empty((T, n))
        for k in range(n):
            log_probs[:, k] = multivariate_normal.logpdf(
                X,
                mean=self._model.means_[k],
                cov=self._model.covars_[k],
            )
        return log_probs

    def _forward_filter(self, X: np.ndarray) -> np.ndarray:
        """
        Forward algorithm: P(state_t | obs_1:t) for every t.

        Operates entirely in log-space for numerical stability.
        Each alpha_t depends only on alpha_{t-1} and obs_t — strictly causal.

        Parameters
        ----------
        X: shape (T, n_features)

        Returns
        -------
        alpha: shape (T, n_states) — normalized filtered probabilities.
        """
        T = len(X)
        n = self._n_states

        log_emissions = self._log_emission_probs(X)                 # (T, n)
        log_transmat = np.log(self._model.transmat_ + 1e-300)       # (n, n)
        log_startprob = np.log(self._model.startprob_ + 1e-300)     # (n,)

        log_alpha = np.empty((T, n))

        # t = 0 : initialization
        log_alpha[0] = log_startprob + log_emissions[0]
        log_alpha[0] -= np.logaddexp.reduce(log_alpha[0])           # normalize

        # t = 1..T-1 : forward pass
        for t in range(1, T):
            # log_alpha[t-1, :, None] + log_transmat  →  shape (n, n)
            # element [j, k] = log_alpha[t-1, j] + log P(state_k | state_j)
            # logaddexp over j (axis=0) → log sum_j(alpha_{t-1,j} * A_{j,k})
            log_pred = np.logaddexp.reduce(
                log_alpha[t - 1, :, np.newaxis] + log_transmat,
                axis=0,
            )                                                        # (n,)
            log_alpha[t] = log_pred + log_emissions[t]
            log_alpha[t] -= np.logaddexp.reduce(log_alpha[t])       # normalize

        return np.exp(log_alpha)

    # ------------------------------------------------------------------
    # Stability filter
    # ------------------------------------------------------------------

    def _reset_stability(self) -> None:
        self._confirmed_regime = -1
        self._consecutive_bars = 0
        self._pending_regime = -1
        self._pending_count = 0
        self._regime_history = []

    def _update_stability(self, raw_state: int) -> None:
        """
        Update pending / confirmed regime tracking for one bar.

        A regime change is "confirmed" only after the new raw_state has
        appeared for stability_bars consecutive bars. Until confirmed,
        is_confirmed remains False and consecutive_bars is not incremented.
        """
        self._regime_history.append(raw_state)

        if raw_state == self._pending_regime:
            self._pending_count += 1
        else:
            self._pending_regime = raw_state
            self._pending_count = 1

        if self._pending_count >= self.config.stability_bars:
            if self._confirmed_regime != raw_state:
                old_label = (
                    self._state_labels[int(self._raw_to_return_rank[self._confirmed_regime])]
                    if self._confirmed_regime >= 0
                    else "NONE"
                )
                new_label = self._state_labels[int(self._raw_to_return_rank[raw_state])]
                logger.warning(
                    "Regime change CONFIRMED: %s → %s", old_label, new_label
                )
                self._consecutive_bars = self._pending_count
            else:
                self._consecutive_bars += 1
            self._confirmed_regime = raw_state
        elif self._confirmed_regime == -1:
            # Haven't confirmed any regime yet; use pending as best guess.
            self._confirmed_regime = raw_state
            self._consecutive_bars = self._pending_count

    def _build_regime_state(
        self,
        raw_state: int,
        posteriors: np.ndarray,
        timestamp: Optional[pd.Timestamp],
    ) -> RegimeState:
        return_rank = int(self._raw_to_return_rank[raw_state])
        info = self._regime_infos.get(return_rank)
        vol_env = info.vol_environment if info else "unknown"
        is_confirmed = self._pending_count >= self.config.stability_bars

        return RegimeState(
            label=self._state_labels[return_rank],
            state_id=return_rank,
            probability=float(posteriors[raw_state]),
            state_probabilities=posteriors.copy(),
            timestamp=timestamp,
            is_confirmed=is_confirmed,
            consecutive_bars=self._consecutive_bars,
            vol_environment=vol_env,
            n_states=self._n_states,
            regime_info=info,
        )

    # ------------------------------------------------------------------
    # Public prediction API
    # ------------------------------------------------------------------

    def predict(self, features: pd.DataFrame) -> RegimeState:
        """
        Predict regime for the most recent bar using the forward algorithm.

        Only data in `features` is used — strictly causal. Safe to call
        bar-by-bar in live trading (runs forward pass over the full window
        each call; cache alpha via predict_regime_filtered for efficiency).
        """
        if not self.is_fitted():
            raise RuntimeError("HMMEngine.fit() must be called before predict().")
        X = features.values.astype(np.float64)
        alpha = self._forward_filter(X)
        posteriors = alpha[-1]
        raw_state = int(np.argmax(posteriors))
        self._update_stability(raw_state)
        ts = features.index[-1] if hasattr(features.index, "__len__") else None
        return self._build_regime_state(raw_state, posteriors, ts)

    def predict_regime_filtered(self, features: pd.DataFrame) -> RegimeState:
        """Alias for predict(). Emphasizes that the forward algorithm is used."""
        return self.predict(features)

    def predict_regime_proba(self, features: pd.DataFrame) -> np.ndarray:
        """
        Return the filtered probability distribution over all states for the
        most recent bar. Shape: (n_states,).
        """
        if not self.is_fitted():
            raise RuntimeError("HMMEngine.fit() must be called before predict_regime_proba().")
        X = features.values.astype(np.float64)
        alpha = self._forward_filter(X)
        return alpha[-1]

    def predict_sequence(self, features: pd.DataFrame) -> list[RegimeState]:
        """
        Return a RegimeState for every bar in `features`.

        Runs one full forward pass (O(T·n²)) then builds RegimeState objects.
        Stability tracking is reset at the start of each call.
        Used by the backtester to process a full test window.
        """
        if not self.is_fitted():
            raise RuntimeError("HMMEngine.fit() must be called before predict_sequence().")
        X = features.values.astype(np.float64)
        alpha = self._forward_filter(X)

        self._reset_stability()
        states: list[RegimeState] = []
        for t in range(len(X)):
            posteriors = alpha[t]
            raw_state = int(np.argmax(posteriors))
            self._update_stability(raw_state)
            ts = features.index[t] if hasattr(features.index, "__len__") else None
            states.append(self._build_regime_state(raw_state, posteriors, ts))

        return states

    # ------------------------------------------------------------------
    # Stability / flicker helpers
    # ------------------------------------------------------------------

    def get_regime_stability(self) -> int:
        """Consecutive bars in the current confirmed regime."""
        return self._consecutive_bars

    def detect_regime_change(self) -> bool:
        """
        Return True if a new regime was confirmed on the most recent bar.

        True only on the first bar where pending_count == stability_bars
        and the confirmed regime just changed.
        """
        if not self._regime_history:
            return False
        recent = self._regime_history[-self.config.stability_bars:]
        if len(recent) < self.config.stability_bars:
            return False
        return (
            len(set(recent)) == 1
            and self._consecutive_bars == self.config.stability_bars
        )

    def get_regime_flicker_rate(self) -> float:
        """Number of regime changes per bar in the last flicker_window bars."""
        window = self._regime_history[-self.config.flicker_window:]
        if len(window) < 2:
            return 0.0
        changes = sum(1 for a, b in zip(window, window[1:]) if a != b)
        return changes / max(len(window) - 1, 1)

    def is_flickering(self) -> bool:
        """True if regime changes exceed flicker_threshold in the recent window."""
        window = self._regime_history[-self.config.flicker_window:]
        if len(window) < 2:
            return False
        changes = sum(1 for a, b in zip(window, window[1:]) if a != b)
        return changes > self.config.flicker_threshold

    def get_transition_matrix(self) -> np.ndarray:
        """Return the learned transition probability matrix (n_states × n_states)."""
        if not self.is_fitted():
            raise RuntimeError("Model not fitted.")
        return self._model.transmat_.copy()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """
        Pickle the engine (model + config + metadata) to `path`.
        Metadata includes n_regimes, bic was logged during fit, training_date,
        and labels — all available via the instance attributes after loading.
        """
        payload = {
            "model": self._model,
            "n_states": self._n_states,
            "state_labels": self._state_labels,
            "return_sort_to_raw": self._return_sort_to_raw,
            "raw_to_return_rank": self._raw_to_return_rank,
            "vol_sort_to_raw": self._vol_sort_to_raw,
            "raw_to_vol_rank": self._raw_to_vol_rank,
            "regime_infos": self._regime_infos,
            "config": self.config,
            "saved_at": datetime.utcnow().isoformat(),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.info("HMMEngine saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "HMMEngine":
        """Load a previously saved HMMEngine from a pickle file."""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        engine = cls(payload["config"])
        engine._model = payload["model"]
        engine._n_states = payload["n_states"]
        engine._state_labels = payload["state_labels"]
        engine._return_sort_to_raw = payload["return_sort_to_raw"]
        engine._raw_to_return_rank = payload["raw_to_return_rank"]
        engine._vol_sort_to_raw = payload["vol_sort_to_raw"]
        engine._raw_to_vol_rank = payload["raw_to_vol_rank"]
        engine._regime_infos = payload["regime_infos"]
        engine._reset_stability()
        logger.info("HMMEngine loaded from %s (n_states=%d)", path, engine._n_states)
        return engine

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def is_fitted(self) -> bool:
        """Return True if the model has been fitted at least once."""
        return self._model is not None

    def get_regime_infos(self) -> dict[int, RegimeInfo]:
        """Return {return_rank: RegimeInfo} for all fitted regimes."""
        return dict(self._regime_infos)
