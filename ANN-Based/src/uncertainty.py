"""
Epistemic uncertainty via Gaussian Discriminant Analysis (per-class Gaussian in feature space), per the paper
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class GDAUncertaintyModel:
    feature_columns: List[str]
    mu: Dict[int, np.ndarray] = field(default_factory=dict)
    sigma_inv: Dict[int, np.ndarray] = field(default_factory=dict)
    log_det_sigma: Dict[int, float] = field(default_factory=dict)
    regularization: float = 1e-6

    def __post_init__(self):
        self.regularization = float(self.regularization)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GDAUncertaintyModel":
        for c in np.unique(y):
            X_c = X[y == c]
            if len(X_c) < X.shape[1] + 1:
                logger.warning(
                    f"Class {c} has only {len(X_c)} samples for "
                    f"{X.shape[1]} features - covariance will be poorly "
                    f"conditioned even with regularization. Consider "
                    f"merging/dropping this class or reducing feature dims."
                )
            mu_c = X_c.mean(axis=0)
            cov_c = np.cov(X_c, rowvar=False)
            cov_c = np.atleast_2d(cov_c)
            cov_c += self.regularization * np.eye(cov_c.shape[0])

            eigvals, _ = np.linalg.eigh(cov_c)
            if np.any(eigvals <= 0):
                logger.warning(
                    f"Class {c}: covariance not positive-definite even "
                    f"after regularization (min eigenvalue={eigvals.min():.2e}). "
                    f"Increase uncertainty.covariance_regularization in "
                    f"configs/default.yaml."
                )

            sign, log_det = np.linalg.slogdet(cov_c)
            if sign <= 0:
                logger.warning(f"Class {c}: non-positive determinant sign={sign}, "
                                f"uncertainty scores for this class will be unreliable.")

            self.mu[c] = mu_c
            self.sigma_inv[c] = np.linalg.pinv(cov_c)
            self.log_det_sigma[c] = log_det
        return self

    def _log_pdf(self, X: np.ndarray, c: int) -> np.ndarray:
        d = X.shape[1]
        diff = X - self.mu[c]
        maha = np.einsum("ij,jk,ik->i", diff, self.sigma_inv[c], diff)
        return -0.5 * (d * np.log(2 * np.pi) + self.log_det_sigma[c] + maha)

    def epistemic_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """
        Returns per-point epistemic uncertainty u_i = -max_c log N(x_i; mu_c, Sigma_c)
        """
        classes = sorted(self.mu.keys())
        log_pdfs = np.stack([self._log_pdf(X, c) for c in classes], axis=1)

        log_pdfs = np.nan_to_num(
            log_pdfs,
            nan=-np.finfo(np.float64).max,
            posinf=np.finfo(np.float64).max,
            neginf=-np.finfo(np.float64).max,
        )

        u = -log_pdfs.max(axis=1)

        n_inf = np.isinf(u).sum()
        n_nan = np.isnan(u).sum()
        if n_inf > 0 or n_nan > 0:
            logger.warning(
                f"epistemic_uncertainty: {n_inf} Inf and {n_nan} NaN values "
                f"in raw uncertainty array after nan_to_num guard - this "
                f"should not happen; increase covariance_regularization "
                f"in configs/default.yaml (current: {self.regularization})."
            )

        return u

    @staticmethod
    def normalize(u: np.ndarray, u_min: float = None, u_max: float = None) -> np.ndarray:
        """Min-max normalize to [0,1]"""
        finite_mask = np.isfinite(u)
        if not finite_mask.all():
            cap = np.percentile(u[finite_mask], 99.9) if finite_mask.any() else 0.0
            u = np.where(finite_mask, u, cap)

        u_min = u.min() if u_min is None else u_min
        u_max = u.max() if u_max is None else u_max
        return (u - u_min) / (u_max - u_min + 1e-12)


def select_gda_feature_columns(all_feature_columns: List[str], config: dict) -> List[str]:
    """Excludes x,y,z from the GDA feature set unless explicitly overridden"""
    include_xyz = config["uncertainty"].get("include_xyz_in_gda", False)
    if include_xyz:
        logger.warning(
            "uncertainty.include_xyz_in_gda=True - epistemic uncertainty will "
            "be contaminated by absolute spatial position (ISSUES.md M-1). "
            "Only enable this for a deliberate ablation."
        )
        return all_feature_columns
    return [c for c in all_feature_columns if c not in ("x", "y", "z")]