# debug_barley.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("Barley.data.csv")

print("="*60)
print("BARLEY DATA DEBUG")
print("="*60)

# Check shape
print(f"\nDataset shape: {df.shape} (rows, columns)")

# Check columns
print(f"\nColumn names (first 10):")
print(df.columns[:10].tolist())

# Check label column
print(f"\nPredictor (label) column unique values: {df['Predictor'].nunique()}")
print(f"Class distribution:")
print(df['Predictor'].value_counts().sort_index())

# Check spectral data
print(f"\nSpectral data (columns 1 to 332):")
X = df.iloc[:, 1:332].values
print(f"Shape: {X.shape}")
print(f"Data type: {X.dtype}")
print(f"Range: [{X.min():.4f}, {X.max():.4f}]")
print(f"Mean: {X.mean():.4f}, Std: {X.std():.4f}")

# Check for missing values
print(f"\nMissing values in spectra: {X.shape[0] - np.sum(~np.isnan(X).any(axis=1))}")

print("\n" + "="*60)
