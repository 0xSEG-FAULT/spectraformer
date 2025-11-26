# visualize_data.py

import argparse
import numpy as np
import pandas as pd
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from data_preprocessing import apply_preprocessing
from dataset_scio import load_scio_csv


def plot_random_spectra(X, y, label_names, n_samples=10, title="Random spectra"):
    """
    Plot n_samples random spectra with different colors.
    """
    idx = np.random.choice(len(X), size=min(n_samples, len(X)), replace=False)
    plt.figure(figsize=(10, 6))
    for i in idx:
        plt.plot(X[i], alpha=0.7, label=label_names[y[i]])
    plt.xlabel("Wavelength index")
    plt.ylabel("Intensity")
    plt.title(title)
    # Avoid huge legends if many classes: only show unique labels of sampled
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=8, loc="best")
    plt.tight_layout()


def plot_mean_spectra_by_class(X, y, label_names, title="Mean spectrum per class"):
    """
    Compute and plot the mean spectrum of each class.
    """
    classes = np.unique(y)
    plt.figure(figsize=(10, 6))
    for c in classes:
        X_c = X[y == c]
        mean_spec = X_c.mean(axis=0)
        plt.plot(mean_spec, label=label_names[c])
    plt.xlabel("Wavelength index")
    plt.ylabel("Mean intensity")
    plt.title(title)
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()


def plot_pca_2d(X, y, label_names, title="PCA (2D) of spectra"):
    """
    Run PCA on spectra and plot the first two principal components,
    colored by class.
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    classes = np.unique(y)
    plt.figure(figsize=(8, 6))
    for c in classes:
        X_c = X_pca[y == c]
        plt.scatter(X_c[:, 0], X_c[:, 1], s=10, label=label_names[c], alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()


def main(args):
    # Load CSV (same loader as training code)
    X, y, le = load_scio_csv(
        args.csv_path,
        label_column=args.label_column,
        wavelength_start_index=args.wavelength_start_index,
        wavelength_end_index=args.wavelength_end_index
    )

    # Optional preprocessing (same pipeline string as train.py, e.g. "SM")
    if args.preprocessing:
        X, _ = apply_preprocessing(X, pipeline=args.preprocessing)

    label_names = le.inverse_transform(np.arange(len(le.classes_)))

    # 1) Random spectra
    plot_random_spectra(X, y, label_names,
                        n_samples=args.n_random,
                        title=f"Random spectra ({args.preprocessing or 'raw'})")

    # 2) Mean spectrum per class
    plot_mean_spectra_by_class(X, y, label_names,
                               title=f"Mean spectrum per class ({args.preprocessing or 'raw'})")

    # 3) PCA 2D
    plot_pca_2d(X, y, label_names,
                title=f"PCA of spectra ({args.preprocessing or 'raw'})")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to dataset CSV (barley, chickpea, sorghum, or combined).")
    parser.add_argument("--label_column", type=str, default="cultivar",
                        help="Name of label column in the CSV.")
    parser.add_argument("--wavelength_start_index", type=int, default=0,
                        help="Index of the first spectral feature column.")
    parser.add_argument("--wavelength_end_index", type=int, default=None,
                        help="Index after last spectral feature column.")
    parser.add_argument("--preprocessing", type=str, default="SM",
                        help="Preprocessing pipeline string (e.g., S, SM, 0M, SA0M). "
                             "Use empty string '' for raw spectra.")
    parser.add_argument("--n_random", type=int, default=10,
                        help="Number of random spectra to plot.")

    args = parser.parse_args()
    # Allow empty preprocessing: if user passes '' we treat as raw
    if args.preprocessing == "''" or args.preprocessing == '""':
        args.preprocessing = ""
    main(args)
