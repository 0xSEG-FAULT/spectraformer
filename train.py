import argparse
import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR

from data_preprocessing import apply_preprocessing
from dataset_scio import load_scio_csv, NIRGrainDataset
from model_spectraformer import Spectraformer


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def eval_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            running_loss += loss.item() * x.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    print(f"Device: {device}")

    # Load best hyperparameters from JSON file
    if os.path.exists(args.params_file):
        print(f"Loading best hyperparameters from {args.params_file}...")
        with open(args.params_file, 'r') as f:
            best_params = json.load(f)
        
        print("Best hyperparameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        
        # Override command-line args with best params
        lr = best_params.get('lr', args.lr)
        batch_size = best_params.get('batch_size', args.batch_size)
        base_channels = best_params.get('base_channels', args.base_channels)
        dropout = best_params.get('dropout', args.dropout)
        weight_decay = best_params.get('weight_decay', args.weight_decay)
        optimizer_name = best_params.get('optimizer', 'SGD')
    else:
        print(f"Warning: {args.params_file} not found. Using command-line arguments.")
        lr = args.lr
        batch_size = args.batch_size
        base_channels = args.base_channels
        dropout = args.dropout
        weight_decay = args.weight_decay
        optimizer_name = args.optimizer

    # Load data
    X, y, le = load_scio_csv(
        args.csv_path,
        label_column=args.label_column,
        wavelength_start_index=args.wavelength_start_index,
        wavelength_end_index=args.wavelength_end_index
    )

    input_length = X.shape[1]
    num_classes = len(np.unique(y))

    print(f"Data loaded: {len(X)} samples, {num_classes} classes, {input_length} features")

    # Preprocessing
    print(f"Applying preprocessing: {args.preprocessing}")
    X_proc, scaler = apply_preprocessing(X, pipeline=args.preprocessing)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_proc, y, test_size=0.3, stratify=y, random_state=args.seed
    )

    train_dataset = NIRGrainDataset(X_train, y_train)
    test_dataset = NIRGrainDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = Spectraformer(
        input_length=input_length,
        num_classes=num_classes,
        base_channels=base_channels,
        dropout=dropout
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_name == "SGD":
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

    best_test_acc = 0.0

    print(f"\nTraining for {args.epochs} epochs...")
    print(f"Optimizer: {optimizer_name}, LR: {lr}, Batch Size: {batch_size}")
    print("="*80)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader,
                                                criterion, optimizer, device)
        test_loss, test_acc = eval_model(model, test_loader,
                                         criterion, device)
        scheduler.step()

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            if args.save_model:
                os.makedirs(args.output_dir, exist_ok=True)
                save_path = os.path.join(args.output_dir, "best_spectraformer.pt")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "label_encoder_classes": le.classes_,
                    "scaler": scaler,
                }, save_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} "
                  f"| Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f} "
                  f"| Best Test Acc: {best_test_acc:.4f}")

    print("="*80)
    print(f"Training completed! Best test accuracy: {best_test_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to dataset CSV")
    parser.add_argument("--label_column", type=str, default="Predictor",
                        help="Name of label column")
    parser.add_argument("--wavelength_start_index", type=int, default=1,
                        help="Index of first spectral feature")
    parser.add_argument("--wavelength_end_index", type=int, default=332,
                        help="Index after last spectral feature")
    parser.add_argument("--preprocessing", type=str, default="SM",
                        help="Preprocessing pipeline (e.g., SM, S, SA0M)")

    # Hyperparameters (defaults, overridden by JSON if file exists)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="SGD", choices=["SGD", "Adam"])

    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")

    # JSON file with best hyperparameters
    parser.add_argument("--params_file", type=str, default="best_hyperparameters_SM.json",
                        help="JSON file with best hyperparameters from tuning")

    # Saving
    parser.add_argument("--save_model", action="store_true", default=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints")

    args = parser.parse_args()
    main(args)
