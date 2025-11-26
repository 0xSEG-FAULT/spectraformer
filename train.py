# train.py

import argparse
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
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

    # 1. Load raw data
    X, y, le = load_scio_csv(
        args.csv_path,
        label_column=args.label_column,
        wavelength_start_index=args.wavelength_start_index,
        wavelength_end_index=args.wavelength_end_index
    )
    input_length = X.shape[1]
    num_classes = len(np.unique(y))

    # 2. Preprocessing
    X_proc, scaler = apply_preprocessing(X, pipeline=args.preprocessing)

    # 3. Train/test split (7:3, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_proc, y, test_size=0.3, stratify=y, random_state=args.seed
    )

    train_dataset = NIRGrainDataset(X_train, y_train)
    test_dataset = NIRGrainDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    # 4. Model
    model = Spectraformer(input_length=input_length,
                          num_classes=num_classes,
                          base_channels=args.base_channels,
                          dropout=args.dropout).to(device)

    # 5. Loss and optimizer (SGD + cross-entropy as in paper)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(),
                    lr=args.lr,
                    momentum=0.9,
                    weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    best_test_acc = 0.0

    # 6. Training loop
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

        print(f"Epoch {epoch:03d} "
              f"| Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} "
              f"| Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f} "
              f"| Best Test Acc: {best_test_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to SCIO dataset CSV.")
    parser.add_argument("--label_column", type=str, default="cultivar",
                        help="Name of the label column.")
    parser.add_argument("--wavelength_start_index", type=int, default=0,
                        help="Index of first spectral feature column in CSV.")
    parser.add_argument("--wavelength_end_index", type=int, default=None,
                        help="Index after last spectral feature column.")

    # Preprocessing (default SM, which is strong & stable)
    parser.add_argument("--preprocessing", type=str, default="SM",
                        help="Preprocessing pipeline string, e.g. S, SM, 0M, SA0M.")

    # Model
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.3)

    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr_step", type=int, default=80)
    parser.add_argument("--lr_gamma", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")

    # Saving
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--output_dir", type=str, default="checkpoints")

    args = parser.parse_args()
    main(args)
