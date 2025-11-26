# data_preprocessing.py

import numpy as np
from scipy.signal import savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import MinMaxScaler


def savitzky_golay_smoothing(X, window_length=11, polyorder=2):
    """
    Apply Savitzky–Golay smoothing along the spectral axis (last dimension).
    X: numpy array of shape (n_samples, n_features)
    """
    if window_length >= X.shape[1]:
        window_length = X.shape[1] - (1 - X.shape[1] % 2)
    if window_length % 2 == 0:
        window_length += 1
    return savgol_filter(X, window_length=window_length,
                         polyorder=polyorder, axis=1)


def _airpls_baseline(y, lam=1e5, p=0.01, niter=15):
    """
    Asymmetric least squares baseline correction for a single spectrum.
    Returns the estimated baseline.
    """
    L = y.shape[0]
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    D = lam * D.dot(D.T)

    w = np.ones(L)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + D
        z = spsolve(Z, w * y)
        d = y - z
        w[d > 0] = p
        w[d <= 0] = 1 - p
    return z


def airpls_baseline_correction(X, lam=1e5, p=0.01, niter=15):
    """
    Apply airPLS baseline correction spectrum-wise.
    """
    corrected = np.zeros_like(X)
    for i in range(X.shape[0]):
        baseline = _airpls_baseline(X[i], lam=lam, p=p, niter=niter)
        corrected[i] = X[i] - baseline
    return corrected


def remove_negative_values(X):
    """
    Set negative values to zero (0 method in paper).
    """
    X = X.copy()
    X[X < 0] = 0.0
    return X


def minmax_normalize(X):
    """
    Min–Max normalization across features to [0, 1].
    Fit scaler on all samples, then transform.
    """
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def apply_preprocessing(X, pipeline="SM"):
    """
    Apply a preprocessing pipeline given by a string of characters:
      S: Savitzky–Golay smoothing
      A: airPLS baseline correction
      0: negative-value removal
      M: Min–Max normalization

    Example pipelines:
      "S"   : Savitzky–Golay
      "SM"  : Savitzky–Golay + Min–Max
      "0M"  : Negative removal + Min–Max
      "SA0M": S + A + 0 + M
    Returns:
      X_proc: processed numpy array
      scaler: MinMaxScaler or None
    """
    X_proc = X.astype(np.float64).copy()
    scaler = None

    for step in pipeline:
        if step == "S":
            X_proc = savitzky_golay_smoothing(X_proc)
        elif step == "A":
            X_proc = airpls_baseline_correction(X_proc)
        elif step == "0":
            X_proc = remove_negative_values(X_proc)
        elif step == "M":
            X_proc, scaler = minmax_normalize(X_proc)
        else:
            raise ValueError(f"Unknown preprocessing step: {step}")

    return X_proc, scaler
