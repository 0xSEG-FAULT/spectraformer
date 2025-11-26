# dataset_scio.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


class NIRGrainDataset(Dataset):
    """
    PyTorch Dataset for SCIO grain NIR spectra.
    """
    def __init__(self, X, y):
        """
        X: numpy array (n_samples, n_features)
        y: numpy array of integer labels (n_samples,)
        """
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        # reshape to (C=1, L)
        x = np.expand_dims(x, axis=0)
        y = self.y[idx]
        return x, y


def load_scio_csv(csv_path,
                  label_column="cultivar",
                  wavelength_start_index=0,
                  wavelength_end_index=None):
    """
    Load SCIO dataset from a CSV file.
    Assumes:
      - spectral features: columns [wavelength_start_index : wavelength_end_index]
      - label: column 'label_column'
    Returns:
      X: numpy array of shape (n_samples, n_features)
      y: integer-encoded labels
      le: fitted LabelEncoder (for decoding)
    """
    df = pd.read_csv(csv_path)

    if wavelength_end_index is None:
        wavelength_end_index = len(df.columns) - 1

    feature_cols = df.columns[wavelength_start_index:wavelength_end_index]
    X = df[feature_cols].values

    labels = df[label_column].values
    le = LabelEncoder()
    y = le.fit_transform(labels)

    return X, y, le
