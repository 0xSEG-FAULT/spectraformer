# tune_hyperparameters.py

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import optuna
from optuna.trial import TrialState

from data_preprocessing import apply_preprocessing
from dataset_scio import load_scio_csv, NIRGrainDataset
from model_spectraformer import Spectraformer


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def objective(trial, X_train, y_train, X_val, y_val, input_length, num_classes, device):
    """
    Optuna objective function for hyperparameter optimization.
    """
    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    base_channels = trial.suggest_categorical("base_channels", [8, 16, 32])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    optimizer_name = trial.suggest_categorical("optimizer", ["SGD", "Adam"])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    
    # Create datasets
    train_dataset = NIRGrainDataset(X_train, y_train)
    val_dataset = NIRGrainDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = Spectraformer(
        input_length=input_length,
        num_classes=num_classes,
        base_channels=base_channels,
        dropout=dropout
    ).to(device)
    
    # Create optimizer
    if optimizer_name == "SGD":
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    criterion = nn.CrossEntropyLoss()
    
    # Train for a few epochs (reduced for faster tuning)
    num_epochs = 20
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                preds = outputs.argmax(dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(y.cpu().numpy())
        
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        val_acc = accuracy_score(all_labels, all_preds)
        
        best_val_acc = max(best_val_acc, val_acc)
        
        # Report intermediate value for pruning
        trial.report(val_acc, epoch)
        
        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return best_val_acc


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    
    print(f"Device: {device}")
    print(f"Starting hyperparameter optimization with {args.n_trials} trials...")
    
    # Load data
    X, y, le = load_scio_csv(
        args.csv_path,
        label_column=args.label_column,
        wavelength_start_index=args.wavelength_start_index,
        wavelength_end_index=args.wavelength_end_index
    )
    
    input_length = X.shape[1]
    num_classes = len(np.unique(y))
    
    print(f"Loaded {len(X)} samples, {num_classes} classes, {input_length} features")
    
    # Preprocessing
    X_proc, _ = apply_preprocessing(X, pipeline=args.preprocessing)
    
    # Split: 70% train, 15% validation (for tuning), 15% test (for final eval)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_proc, y, test_size=0.3, stratify=y, random_state=args.seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=args.seed
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Create Optuna study
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val,
                               input_length, num_classes, device),
        n_trials=args.n_trials,
        timeout=args.timeout
    )
    
    # Print results
    print("\n" + "="*60)
    print("Hyperparameter Optimization Complete!")
    print("="*60)
    
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"  Pruned trials: {len(pruned_trials)}")
    print(f"  Completed trials: {len(complete_trials)}")
    
    print("\nBest trial:")
    best_trial = study.best_trial
    print(f"  Value (validation accuracy): {best_trial.value:.4f}")
    print("  Params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    
    # Save best hyperparameters
    import json
    best_params_file = f"best_hyperparameters_{args.preprocessing}.json"
    with open(best_params_file, 'w') as f:
        json.dump(best_trial.params, f, indent=2)
    print(f"\nBest hyperparameters saved to {best_params_file}")
    
    # Show command to run training with best params
    print("\nRun training with best hyperparameters:")
    print(f"python train.py --csv_path {args.csv_path} "
          f"--label_column {args.label_column} "
          f"--wavelength_start_index {args.wavelength_start_index} "
          f"--wavelength_end_index {args.wavelength_end_index} "
          f"--preprocessing {args.preprocessing} "
          f"--lr {best_trial.params['lr']:.6f} "
          f"--batch_size {best_trial.params['batch_size']} "
          f"--base_channels {best_trial.params['base_channels']} "
          f"--dropout {best_trial.params['dropout']:.4f} "
          f"--weight_decay {best_trial.params['weight_decay']:.6f} "
          f"--epochs 200 --save_model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--label_column", type=str, default="Predictor")
    parser.add_argument("--wavelength_start_index", type=int, default=1)
    parser.add_argument("--wavelength_end_index", type=int, default=332)
    parser.add_argument("--preprocessing", type=str, default="SM")
    
    # Optuna settings
    parser.add_argument("--n_trials", type=int, default=50,
                        help="Number of Optuna trials (default 50)")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Stop study after this many seconds (optional)")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    
    args = parser.parse_args()
    main(args)
