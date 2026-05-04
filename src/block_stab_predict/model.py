"""Multi-output Random Forest regressor for block-size and stabilising-step prediction.

Wraps :class:`sklearn.ensemble.RandomForestRegressor` with a convenience
API for saving, loading, and inspecting feature importance.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor  # type: ignore[import-untyped]

from block_stab_predict.schema import RFConfig


class BlockStabPredictor:
    """Multi-output Random Forest for predicting (block_size, max_stab_step).

    Args:
        config: Predictor configuration.
    """

    def __init__(self, config: RFConfig) -> None:
        self.config = config
        self._model = RandomForestRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_leaf=config.min_samples_leaf,
            random_state=config.random_state,
            n_jobs=config.n_jobs,
        )
        self._feature_names: list[str] | None = None

    # ── Training ───────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> BlockStabPredictor:
        """Train the random forest on pre-built feature and target matrices.

        ``X`` and ``Y`` are typically obtained from
        :func:`~block_stab_predict.dataset.build_X_y` or
        :func:`~block_stab_predict.dataset.train_test_split_by_sample`.

        Args:
            X: float array of shape ``(n_examples, n_features)``, produced
               by :func:`~block_stab_predict.features.compute_features`.
            Y: float array of shape ``(n_examples, n_targets)`` where
               column 0 is ``block_size`` and column 1 is
               ``max_stab_step`` (ordered per ``config.target_fields``).
            feature_names:
                Optional column labels for *X*, stored for later
                inspection via :meth:`feature_importances`.  When
                provided, the length must match ``X.shape[1]``.

        Returns:
            self (fitted estimator).
        """
        self._model.fit(X, Y)
        if feature_names is not None and len(feature_names) == X.shape[1]:
            self._feature_names = list(feature_names)
        return self

    # ── Inference ──────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict (block_size, max_stab_step) for each row in *X*.

        Args:
            X: float array of shape ``(n_examples, n_features)`` with
               the same feature columns used during :meth:`fit`.

        Returns:
            float array of shape ``(n_examples, 2)`` with columns
            ``[block_size, max_stab_step]``.
        """
        return self._model.predict(X)  # type: ignore[no-any-return]

    # ── Feature importance ─────────────────────────────────────────────

    def feature_importances(self) -> tuple[np.ndarray, list[str]]:
        """Return feature importance scores with corresponding names.

        Returns:
            ``(importances, names)`` sorted in descending order of
            importance.  If feature names were not provided at fit time,
            ``names`` will be ``["feat_0", "feat_1", ...]``.
        """
        n_features = self._model.n_features_in_
        if self._feature_names is not None and len(self._feature_names) == n_features:
            names = self._feature_names
        else:
            names = [f"feat_{i}" for i in range(n_features)]

        importances = self._model.feature_importances_  # type: ignore[union-attr]
        order = np.argsort(importances)[::-1]
        return importances[order], [names[i] for i in order]

    # ── Persistence ────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Persist the fitted predictor to disk.

        Args:
            path: Destination path (conventionally ``*.joblib``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> BlockStabPredictor:
        """Load a :class:`BlockStabPredictor` saved with :meth:`save`.

        Args:
            path: Path to the ``.joblib`` file.

        Returns:
            A ready-to-use predictor.
        """
        return joblib.load(path)  # type: ignore[no-any-return]
