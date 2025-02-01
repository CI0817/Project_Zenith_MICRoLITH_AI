import argparse
import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

# Local imports (must exist in your project)
from dataset import load_oxiod_dataset, load_euroc_mav_dataset, load_dataset_6d_quat
from model import LSTM_PROPERTIES, TCNBLOCK_PROPERTIES, create_model, CustomMultiLossLayer


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for dataset choice and output model name."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset", 
        choices=["oxiod", "euroc"],
        help="Training dataset name (oxiod or euroc)."
    )
    parser.add_argument(
        "output",
        help="Name (without extension) for the saved model checkpoint."
    )
    parser.add_argument(
        "--model_type", 
        choices=["lstm", "tcn", "transformer"],
        default="lstm", 
        help="Model architecture."
    )
    return parser.parse_args()


def build_file_lists(dataset_choice: str) -> Tuple[List[str], List[str]]:
    """
    Returns lists of IMU and ground-truth filenames depending on the chosen dataset.
    """
    if dataset_choice == "oxiod":
        imu_files = [
            "Oxford Inertial Odometry Dataset/handheld/data5/syn/imu3.csv",
        ]
        gt_files = [
            "Oxford Inertial Odometry Dataset/handheld/data5/syn/vi3.csv",
        ]
    else:  # 'euroc'
        imu_files = [
            "MH_01_easy/mav0/imu0/data.csv",
            "MH_03_medium/mav0/imu0/data.csv",
            "MH_04_difficult/mav0/imu0/data.csv",
            "V1_01_easy/mav0/imu0/data.csv",
            "V1_03_difficult/mav0/imu0/data.csv",
            "MH_02_easy/mav0/imu0/data.csv",
            "MH_05_difficult/mav0/imu0/data.csv",
            "V1_02_medium/mav0/imu0/data.csv",
        ]
        gt_files = [
            "MH_01_easy/mav0/state_groundtruth_estimate0/data.csv",
            "MH_03_medium/mav0/state_groundtruth_estimate0/data.csv",
            "MH_04_difficult/mav0/state_groundtruth_estimate0/data.csv",
            "V1_01_easy/mav0/state_groundtruth_estimate0/data.csv",
            "V1_03_difficult/mav0/state_groundtruth_estimate0/data.csv",
            "MH_02_easy/mav0/state_groundtruth_estimate0/data.csv",
            "MH_05_difficult/mav0/state_groundtruth_estimate0/data.csv",
            "V1_02_medium/mav0/state_groundtruth_estimate0/data.csv",
        ]
    return imu_files, gt_files


def load_and_process_data(
    dataset_choice: str,
    imu_files: List[str],
    gt_files: List[str],
    window_size: int = 200,
    stride: int = 10,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    Loads multiple sequences from either the OXIOD or EuRoC dataset,
    converting them to windowed IMU data + delta poses.
    Returns a list of sequence dicts and a list of sequence lengths.
    """
    sequence_data = []
    sequence_lengths = []

    for imu_file, gt_file in zip(imu_files, gt_files):
        # 1. Load raw data based on dataset type.
        if dataset_choice == "oxiod":
            gyro_data, acc_data, pos_data, ori_data = load_oxiod_dataset(imu_file, gt_file)
        else:
            gyro_data, acc_data, pos_data, ori_data = load_euroc_mav_dataset(imu_file, gt_file)

        # 2. Create windowed 6D sequences (position and quaternion)
        [x_gyro, x_acc], [y_delta_p, y_delta_q], _, _ = load_dataset_6d_quat(
            gyro_data, acc_data, pos_data, ori_data, window_size=window_size, stride=stride
        )

        sequence_data.append({
            "x_gyro": x_gyro,
            "x_acc": x_acc,
            "y_delta_p": y_delta_p,
            "y_delta_q": y_delta_q,
        })
        sequence_lengths.append(len(x_gyro))

    return sequence_data, sequence_lengths


def split_sequences(
    sequence_data: List[Dict[str, Any]], train_ratio: float = 0.7
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Splits loaded sequences into training and validation sets.
    """
    num_sequences = len(sequence_data)
    split_idx = int(train_ratio * num_sequences)
    return sequence_data[:split_idx], sequence_data[split_idx:]


def concat_and_build_dataset(sequences: List[Dict[str, Any]]) -> TensorDataset:
    """
    Stacks sequence arrays vertically and converts them to a single TensorDataset.
    """
    x_gyro = np.vstack([seq["x_gyro"] for seq in sequences])
    x_acc = np.vstack([seq["x_acc"] for seq in sequences])
    y_delta_p = np.vstack([seq["y_delta_p"] for seq in sequences])
    y_delta_q = np.vstack([seq["y_delta_q"] for seq in sequences])

    # Convert to torch tensors.
    x_gyro = torch.FloatTensor(x_gyro)
    x_acc = torch.FloatTensor(x_acc)
    y_delta_p = torch.FloatTensor(y_delta_p)
    y_delta_q = torch.FloatTensor(y_delta_q)

    return TensorDataset(x_gyro, x_acc, y_delta_p, y_delta_q)


def create_dataloaders(
    train_dataset: TensorDataset, val_dataset: TensorDataset, batch_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates DataLoaders for training and validation datasets.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def quaternion_angle_error(q_true: torch.Tensor, q_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute the angular difference (in radians) between two quaternions.
    """
    q_pred_norm = F.normalize(q_pred, p=2, dim=-1)  # Ensure unit norm
    dot = torch.sum(q_true * q_pred_norm, dim=-1)
    dot = torch.clamp(dot, -1.0, 1.0)
    angles = 2.0 * torch.acos(torch.abs(dot))
    return angles


def write_csv_header(csv_path: Path, hyperparams: Dict[str, Any]) -> None:
    """Write the CSV header if the file does not already exist."""
    if not csv_path.exists():
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            header = [
                "epoch",
                "train_loss",
                "val_loss",
                "val_pos_rmse",
                "val_quat_angle_deg",
                "epoch_time",
                "current_lr",
            ]
            # Append hyperparameter names.
            header.extend(list(hyperparams.keys()))
            writer.writerow(header)


def append_csv_row(csv_path: Path, row: List[Any]) -> None:
    """Append a single row to the CSV file."""
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 100,
    scheduler: Any = None,
    checkpoint_path: str = "checkpoint.pt",
    hyperparams: Dict[str, Any] = None,
) -> Dict[str, List[float]]:
    """
    Main training loop.
    Logs metrics per epoch into a CSV file and saves the best model checkpoint.
    """
    if hyperparams is None:
        hyperparams = {}

    # Prepare CSV log file.
    timestamp = hyperparams.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
    model_type = hyperparams.get("model_type", "lstm")
    csv_filename = f"{model_type}_{timestamp}.csv"
    csv_path = Path(csv_filename).resolve()
    write_csv_header(csv_path, hyperparams)

    best_val_loss = float("inf")
    train_losses, val_losses = [], []
    val_pos_rmse_list, val_quat_angle_deg_list = [], []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for (x_gyro, x_acc, y_delta_p, y_delta_q) in train_loader:
            x_gyro, x_acc = x_gyro.to(device), x_acc.to(device)
            y_delta_p, y_delta_q = y_delta_p.to(device), y_delta_q.to(device)

            optimizer.zero_grad()
            pos_pred, quat_pred = model(x_gyro, x_acc)
            loss = criterion([y_delta_p, y_delta_q], [pos_pred, quat_pred])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)

        # Validation phase.
        model.eval()
        val_loss = 0.0
        pos_mse_total = 0.0
        quat_angle_total = 0.0
        batch_count = 0

        with torch.no_grad():
            for (x_gyro, x_acc, y_delta_p, y_delta_q) in val_loader:
                x_gyro, x_acc = x_gyro.to(device), x_acc.to(device)
                y_delta_p, y_delta_q = y_delta_p.to(device), y_delta_q.to(device)

                pos_pred, quat_pred = model(x_gyro, x_acc)
                batch_loss = criterion([y_delta_p, y_delta_q], [pos_pred, quat_pred])
                val_loss += batch_loss.item()

                batch_pos_mse = F.mse_loss(pos_pred, y_delta_p, reduction="mean")
                pos_mse_total += batch_pos_mse.item()

                batch_angles = quaternion_angle_error(y_delta_q, quat_pred)
                quat_angle_total += batch_angles.mean().item()

                batch_count += 1

        val_loss /= batch_count
        val_losses.append(val_loss)

        mean_pos_rmse = math.sqrt(pos_mse_total / batch_count)
        val_pos_rmse_list.append(mean_pos_rmse)
        mean_angle_deg = (quat_angle_total / batch_count) * 180.0 / math.pi
        val_quat_angle_deg_list.append(mean_angle_deg)

        # Print progress.
        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {epoch_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Pos RMSE: {mean_pos_rmse:.4f} | "
            f"Angle (deg): {mean_angle_deg:.4f}"
        )

        # Step the scheduler after validation.
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        # Save best checkpoint.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_val_loss,
            }, checkpoint_path)

        # Log epoch metrics.
        epoch_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [
            epoch + 1,
            epoch_loss,
            val_loss,
            mean_pos_rmse,
            mean_angle_deg,
            epoch_time_str,
            current_lr,
        ]
        # Append hyperparameter values.
        row.extend([hyperparams[k] for k in hyperparams])
        append_csv_row(csv_path, row)

    return {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_pos_rmse": val_pos_rmse_list,
        "val_quat_angle_deg": val_quat_angle_deg_list,
    }


def main() -> None:
    """
    Main entry point for training:
      1. Parse arguments.
      2. Load and process data.
      3. Split into train/val sets.
      4. Create model, criterion, optimizer, and scheduler.
      5. Train, validate, and save the best checkpoint.
      6. Log each epoch’s metrics in CSV.
    """
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    # Hyperparameters and configuration.
    model_type = args.model_type
    window_size = 200
    stride = 10
    batch_size = 32
    num_epochs = 100
    learning_rate = 1e-3
    scheduler_type = "MultiStepLR"  # "MultiStepLR" or "ReduceLROnPlateau"

    # Scheduler settings.
    scheduler_props = {}
    multi_step_props = {"type": "MultiStepLR", "milestones": [20, 50, 80], "gamma": 0.1}
    reduce_lr_props = {
        "type": "ReduceLROnPlateau",
        "mode": "min",
        "factor": 0.1,
        "patience": 10,
        "threshold": 0.0001,
        "min_lr": 1e-6,
    }
    scheduler_props = multi_step_props if scheduler_type == "MultiStepLR" else reduce_lr_props

    # Print model-specific parameters.
    if model_type == "lstm":
        print(f"LSTM layers: {LSTM_PROPERTIES['num_layers']}, "
              f"hidden size: {LSTM_PROPERTIES['hidden_size']}, "
              f"bidirectional: {LSTM_PROPERTIES['bidirectional']}, "
              f"dropout: {LSTM_PROPERTIES['dropout']}")
    elif model_type == "tcn":
        print(f"TCN layers: {TCNBLOCK_PROPERTIES['num_layers']}, "
              f"kernel size: {TCNBLOCK_PROPERTIES['kernel_size']}, "
              f"dropout: {TCNBLOCK_PROPERTIES['dropout']}")

    print(f"window_size: {window_size}, stride: {stride}, batch_size: {batch_size}, "
          f"num_epochs: {num_epochs}, learning_rate: {learning_rate}")
    if scheduler_type == "MultiStepLR":
        print(f"LR Scheduler: {scheduler_type}, milestones: {scheduler_props['milestones']}, "
              f"gamma: {scheduler_props['gamma']}")
    else:
        print(f"LR Scheduler: {scheduler_type}, mode: {scheduler_props['mode']}, "
              f"factor: {scheduler_props['factor']}, patience: {scheduler_props['patience']}, "
              f"threshold: {scheduler_props['threshold']}, min_lr: {scheduler_props['min_lr']}")

    # Build file lists.
    imu_files, gt_files = build_file_lists(args.dataset)

    # Load and process sequences.
    sequence_data, sequence_lengths = load_and_process_data(
        dataset_choice=args.dataset,
        imu_files=imu_files,
        gt_files=gt_files,
        window_size=window_size,
        stride=stride,
    )
    train_sequences, val_sequences = split_sequences(sequence_data, train_ratio=0.7)

    print(f"\nTotal sequences: {len(sequence_data)}")
    print(f"Training sequences: {len(train_sequences)}")
    print(f"Validation sequences: {len(val_sequences)}\n")

    # Print sequence details.
    for i, seq in enumerate(train_sequences):
        print(f"  Training Sequence {i} has {len(seq['x_gyro'])} windows")
    for i, seq in enumerate(val_sequences):
        print(f"  Validation Sequence {i} has {len(seq['x_gyro'])} windows")

    # Combine sequences to form complete datasets.
    train_dataset = concat_and_build_dataset(train_sequences)
    val_dataset = concat_and_build_dataset(val_sequences)
    print(f"\nTotal training samples: {len(train_dataset)}")
    print(f"Total validation samples: {len(val_dataset)}\n")

    # Create DataLoaders.
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, batch_size=batch_size)

    # Create model.
    print(f"Creating model: {model_type}...")
    model = create_model(window_size=window_size, model_type=model_type)
    model.to(device)

    # Criterion, optimizer, and scheduler.
    criterion = CustomMultiLossLayer(nb_outputs=2)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    if scheduler_props["type"] == "MultiStepLR":
        scheduler = MultiStepLR(optimizer, milestones=scheduler_props["milestones"], gamma=scheduler_props["gamma"])
    else:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=scheduler_props["mode"],
            factor=scheduler_props["factor"],
            patience=scheduler_props["patience"],
            threshold=scheduler_props["threshold"],
            min_lr=scheduler_props["min_lr"],
        )

    # Build hyperparameters dictionary.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = Path(f"{args.output}_{timestamp}.pt")
    hyperparams = {
        "model_type": model_type,
        "sequence_length": window_size,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "lr_scheduler": [f"{k}={v}" for k, v in scheduler_props.items()],
        "optimizer": "Adam",
        "timestamp": timestamp,
    }
    if model_type == "lstm":
        hyperparams.update(LSTM_PROPERTIES)
    elif model_type == "tcn":
        hyperparams.update(TCNBLOCK_PROPERTIES)

    # Train the model.
    metrics = train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        scheduler=scheduler,
        checkpoint_path=str(checkpoint_path),
        hyperparams=hyperparams,
    )


if __name__ == "__main__":
    main()


# """
# This script trains a neural network model on either the Oxford Inertial Odometry

# Usage Example:
#     python train.py dataset_name output_model_name --model_type lstm
# """

# import argparse
# from pathlib import Path
# from datetime import datetime
# import math
# import torch.nn.functional as F
# import csv

# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, TensorDataset
# from torch.optim import Adam
# from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau

# from model import LSTM_PROPERTIES, TCNBLOCK_PROPERTIES

# # ------------------------------------------------------------------------------------------------------------

# # Local imports (these must exist in your project)
# from dataset import (load_oxiod_dataset, load_euroc_mav_dataset, 
#                      load_dataset_6d_quat)
# from model import create_model, CustomMultiLossLayer


# def parse_arguments():
#     """Parse command-line arguments for dataset choice and output model name."""
#     parser = argparse.ArgumentParser()
#     parser.add_argument('dataset', choices=['oxiod', 'euroc'],
#                         help='Training dataset name (oxiod or euroc).')
#     parser.add_argument('output',
#                         help='Name (without extension) for the saved model checkpoint.')
#     parser.add_argument('--model_type', choices=['lstm', 'tcn', 'transformer'],
#                         default='lstm', help='Model architecture.')
#     args = parser.parse_args()
#     return args


# def build_file_lists(dataset_choice):
#     """
#     Returns lists of IMU and GT filenames depending on the chosen dataset.
#     Modify or expand these lists as you add more files.
#     """
#     if dataset_choice == 'oxiod':
#         imu_data_filenames = [
#             'Oxford Inertial Odometry Dataset/handheld/data5/syn/imu3.csv',
#         ]
#         gt_data_filenames = [
#             'Oxford Inertial Odometry Dataset/handheld/data5/syn/vi3.csv',
#         ]
#     else:  # 'euroc'
#         imu_data_filenames = [
#             'MH_01_easy/mav0/imu0/data.csv',
#             'MH_03_medium/mav0/imu0/data.csv',
#             'MH_04_difficult/mav0/imu0/data.csv',
#             'V1_01_easy/mav0/imu0/data.csv',
#             'V1_03_difficult/mav0/imu0/data.csv',
#             'MH_02_easy/mav0/imu0/data.csv',
#             'MH_05_difficult/mav0/imu0/data.csv',
#             'V1_02_medium/mav0/imu0/data.csv'
#         ]
#         gt_data_filenames = [
#             'MH_01_easy/mav0/state_groundtruth_estimate0/data.csv',
#             'MH_03_medium/mav0/state_groundtruth_estimate0/data.csv',
#             'MH_04_difficult/mav0/state_groundtruth_estimate0/data.csv',
#             'V1_01_easy/mav0/state_groundtruth_estimate0/data.csv',
#             'V1_03_difficult/mav0/state_groundtruth_estimate0/data.csv',
#             'MH_02_easy/mav0/state_groundtruth_estimate0/data.csv',
#             'MH_05_difficult/mav0/state_groundtruth_estimate0/data.csv',
#             'V1_02_medium/mav0/state_groundtruth_estimate0/data.csv'
#         ]
#     return imu_data_filenames, gt_data_filenames


# def load_and_process_data(dataset_choice, imu_files, gt_files,
#                           window_size=200, stride=10):
#     """
#     Loads multiple sequences from either OXIOD or EuRoC dataset,
#     then processes each into windowed IMU data + delta poses.
#     Returns a list of dicts, one per sequence, plus a list of lengths.
#     """
#     sequence_data = []
#     sequence_lengths = []

#     for imu_file, gt_file in zip(imu_files, gt_files):
#         # 1. Load raw data
#         if dataset_choice == 'oxiod':
#             gyro_data, acc_data, pos_data, ori_data = load_oxiod_dataset(
#                 imu_file, gt_file
#             )
#         else:  # 'euroc'
#             gyro_data, acc_data, pos_data, ori_data = load_euroc_mav_dataset(
#                 imu_file, gt_file
#             )

#         # 2. Convert to 6D windowed sequences (position + quaternion)
#         [x_gyro, x_acc], [y_delta_p, y_delta_q], _, _ = load_dataset_6d_quat(
#             gyro_data, acc_data, pos_data, ori_data,
#             window_size=window_size, stride=stride
#         )

#         sequence_data.append({
#             'x_gyro': x_gyro,
#             'x_acc': x_acc,
#             'y_delta_p': y_delta_p,
#             'y_delta_q': y_delta_q
#         })
#         sequence_lengths.append(len(x_gyro))

#     return sequence_data, sequence_lengths


# def split_sequences(sequence_data, train_ratio=0.7):
#     """
#     Splits the loaded sequences into a train set and validation set based on
#     the specified ratio (e.g., 70% for training, 30% for validation).
#     Returns two lists: train_sequences, val_sequences.
#     """
#     num_sequences = len(sequence_data)
#     train_seq_idx = int(train_ratio * num_sequences)

#     train_sequences = sequence_data[:train_seq_idx]
#     val_sequences = sequence_data[train_seq_idx:]
#     return train_sequences, val_sequences


# def concat_and_build_dataset(sequences):
#     """
#     Takes a list of sequence dicts and stacks them vertically (np.vstack)
#     to create a single TensorDataset.
#     """
#     # Combine each field across sequences
#     x_gyro = np.vstack([seq['x_gyro'] for seq in sequences])
#     x_acc = np.vstack([seq['x_acc'] for seq in sequences])
#     y_delta_p = np.vstack([seq['y_delta_p'] for seq in sequences])
#     y_delta_q = np.vstack([seq['y_delta_q'] for seq in sequences])

#     # Convert to torch Tensors
#     x_gyro = torch.FloatTensor(x_gyro)
#     x_acc = torch.FloatTensor(x_acc)
#     y_delta_p = torch.FloatTensor(y_delta_p)
#     y_delta_q = torch.FloatTensor(y_delta_q)

#     return TensorDataset(x_gyro, x_acc, y_delta_p, y_delta_q)


# def create_dataloaders(train_dataset, val_dataset, batch_size=32):
#     """
#     Given train and val TensorDatasets, return corresponding DataLoaders.
#     shuffle=False is used if you want to preserve sequence order within each dataset.
#     """
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#     return train_loader, val_loader


# def quaternion_angle_error(q_true, q_pred):
#     """
#     Compute the angular difference (in radians) between two quaternions.
#     q_true, q_pred: (batch_size, 4) shape
#     """
#     q_pred_norm = F.normalize(q_pred, p=2, dim=-1)  # ensure unit norm
#     dot = (q_true * q_pred_norm).sum(dim=-1)        # dot product
#     dot = torch.clamp(dot, -1.0, 1.0)               # clamp for safety
#     angles = 2.0 * torch.acos(dot.abs())            # angle between them
#     return angles  # shape (batch_size,4)


# def train_loop(model, 
#                train_loader, 
#                val_loader, 
#                criterion, 
#                optimizer,
#                device, 
#                num_epochs=100, 
#                scheduler=None,
#                checkpoint_path='checkpoint.pt',
#                hyperparams=None):
#     """
#     Main training/validation loop.

#     Logs each epoch into a CSV with:
#       - epoch
#       - train_loss
#       - val_loss
#       - val_pos_rmse
#       - val_quat_angle_deg
#       - epoch_time (the current date/time)
#       - current_lr (the learning rate used in this epoch)
#       - plus all items in 'hyperparams' (e.g., # LSTM layers, hidden size, etc.)
#     """

#     if hyperparams is None:
#         hyperparams = {}

#     # We'll build a CSV filename from the model type or any identifier in hyperparams
#     training_start_ts = hyperparams.get('timestamp', datetime.now().strftime('%Y%m%d_%H%M%S'))
#     model_type = hyperparams.get('model_type', 'lstm')
    
#     csv_filename = f"{model_type}_{training_start_ts}.csv"
#     csv_path = Path(csv_filename).resolve()

#     # If CSV doesn't exist, write the header
#     if not csv_path.exists():
#         with open(csv_path, mode='w', newline='') as f:
#             writer = csv.writer(f)
#             # Basic columns
#             header = [
#                 'epoch',
#                 'train_loss',
#                 'val_loss',
#                 'val_pos_rmse',
#                 'val_quat_angle_deg',
#                 'epoch_time',
#                 'current_lr'  # Will store the LR for this epoch
#             ]
#             # Then store hyperparams columns
#             for hp_key in hyperparams:
#                 header.append(hp_key)
#             writer.writerow(header)

#     best_val_loss = float('inf')

#     # For plotting or analysis after training
#     train_losses = []
#     val_losses = []
#     val_pos_rmse_list = []
#     val_quat_angle_deg_list = []

#     for epoch in range(num_epochs):
#         # ------------------ TRAIN PHASE ------------------
#         model.train()
#         epoch_loss = 0.0

#         for (x_gyro, x_acc, y_delta_p, y_delta_q) in train_loader:
#             x_gyro, x_acc = x_gyro.to(device), x_acc.to(device)
#             y_delta_p, y_delta_q = y_delta_p.to(device), y_delta_q.to(device)

#             optimizer.zero_grad()
#             pos_pred, quat_pred = model(x_gyro, x_acc)
#             loss = criterion([y_delta_p, y_delta_q], [pos_pred, quat_pred])
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item()

#         epoch_loss /= len(train_loader)
#         train_losses.append(epoch_loss)

#         # ------------------ VALIDATION PHASE ------------------
#         model.eval()
#         val_loss = 0.0
#         val_pos_mse_sum = 0.0
#         val_quat_angle_sum = 0.0
#         val_batches = 0

#         with torch.no_grad():
#             for (x_gyro, x_acc, y_delta_p, y_delta_q) in val_loader:
#                 x_gyro, x_acc = x_gyro.to(device), x_acc.to(device)
#                 y_delta_p, y_delta_q = y_delta_p.to(device), y_delta_q.to(device)

#                 pos_pred, quat_pred = model(x_gyro, x_acc)
#                 batch_loss = criterion([y_delta_p, y_delta_q], [pos_pred, quat_pred])
#                 val_loss += batch_loss.item()

#                 # Position MSE
#                 batch_pos_mse = F.mse_loss(pos_pred, y_delta_p, reduction='mean')
#                 val_pos_mse_sum += batch_pos_mse.item()

#                 # Quaternion angle difference
#                 batch_angles = quaternion_angle_error(y_delta_q, quat_pred)
#                 val_quat_angle_sum += batch_angles.mean().item()

#                 val_batches += 1

#         val_loss /= val_batches
#         val_losses.append(val_loss)

#         mean_pos_mse = val_pos_mse_sum / val_batches
#         mean_pos_rmse = math.sqrt(mean_pos_mse)
#         val_pos_rmse_list.append(mean_pos_rmse)

#         mean_angle_radians = val_quat_angle_sum / val_batches
#         mean_angle_degrees = (mean_angle_radians * 180.0) / math.pi
#         val_quat_angle_deg_list.append(mean_angle_degrees)

#         # Print to terminal
#         print(f"Epoch {epoch+1}/{num_epochs} | "
#               f"Train Loss: {epoch_loss:.4f} | "
#               f"Val Loss: {val_loss:.4f} | "
#               f"Pos RMSE: {mean_pos_rmse:.4f} | "
#               f"Angle (deg): {mean_angle_degrees:.4f}")

#         # Step the scheduler *after* validating
#         if scheduler is not None:
#             if isinstance(scheduler, ReduceLROnPlateau):
#                 scheduler.step(val_loss)
#             else: # MultiStepLR
#                 scheduler.step()

#         # Check the new/current LR from the optimizer
#         current_lr = optimizer.param_groups[0]['lr']

#         # Save best checkpoint
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'loss': best_val_loss,
#             }, checkpoint_path)

#         # Store a row in the CSV
#         epoch_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         with open(csv_path, mode='a', newline='') as f:
#             writer = csv.writer(f)
#             row = [
#                 epoch + 1,
#                 epoch_loss,
#                 val_loss,
#                 mean_pos_rmse,
#                 mean_angle_degrees,
#                 epoch_time_str,
#                 current_lr
#             ]
#             for hp_key in hyperparams:
#                 row.append(hyperparams[hp_key])
#             writer.writerow(row)

#     # Return metrics for further analysis
#     return {
#         'train_loss': train_losses,
#         'val_loss': val_losses,
#         'val_pos_rmse': val_pos_rmse_list,
#         'val_quat_angle_deg': val_quat_angle_deg_list
#     }


# def main():
#     """
#     Main entry point for training:
#     1. Parse args
#     2. Load and process data
#     3. Split into train/val sets
#     4. Create model, criterion, optimizer
#     5. Train, validate, and save best checkpoint
#     6. Plot training history
#     7. Each epoch's metrics + hyperparams are stored in CSV
#     """
#     args = parse_arguments()

#     # Setup environment
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     torch.manual_seed(0)

#     # -------------------------------------
#     model_type = args.model_type
#     # Hyperparameters
#     window_size = 200
#     stride = 10
#     batch_size = 32 # 32
#     num_epochs = 100
#     learning_rate = 1e-3
#     scheduler_type = 'ReduceLROnPlateau' # 'MultiStepLR' # 'ReduceLROnPlateau'

#     # -------------------------------------
#     # Scheduler properties
#     scheduler_props = {}
#     multiStepLR_props = {
#         'type': 'MultiStepLR',
#         'milestones': [20,50,80],
#         'gamma': 0.1
#     }
#     reduceLROnPlateau_props = {
#         'type': 'ReduceLROnPlateau',
#         'mode': 'min',
#         'factor': 0.1,
#         'patience': 10,
#         'threshold': 0.0001,
#         'min_lr': 1e-6
#     }
#     if scheduler_type == 'MultiStepLR':
#         scheduler_props = multiStepLR_props
#     elif scheduler_type == 'ReduceLROnPlateau':
#         scheduler_props = reduceLROnPlateau_props

#     # -------------------------------------
#     if model_type == 'lstm':
#         # LSTM parameters
#         lstm_layers = LSTM_PROPERTIES['num_layers']
#         lstm_hidden_size = LSTM_PROPERTIES['hidden_size']
#         bidirectional = LSTM_PROPERTIES['bidirectional']
#         dropout_rate = LSTM_PROPERTIES['dropout']
#     elif model_type == 'tcn':
#         # TCN parameters
#         tcn_layers = TCNBLOCK_PROPERTIES['num_layers']
#         tcn_kernel_size = TCNBLOCK_PROPERTIES['kernel_size']
#         dropout_rate = TCNBLOCK_PROPERTIES['dropout']

#     # -------------------------------------
#     lr_scheduler_props = [f"{k}={v}" for k, v in scheduler_props.items()]
#     optimizer_name = "Adam"

#     # Build file lists based on dataset choice
#     imu_files, gt_files = build_file_lists(args.dataset)

#     # Load sequences
#     sequence_data, sequence_lengths = load_and_process_data(
#         dataset_choice=args.dataset,
#         imu_files=imu_files,
#         gt_files=gt_files,
#         window_size=window_size,
#         stride=stride
#     )

#     # Split into train and val
#     train_sequences, val_sequences = split_sequences(sequence_data, train_ratio=0.7)
#     print(f"\nTotal sequences: {len(sequence_data)}")
#     print(f"Training sequences: {len(train_sequences)}")
#     print(f"Validation sequences: {len(val_sequences)}\n")

#     # Print details about each subset
#     print("Training Sequences:")
#     for i, seq in enumerate(train_sequences):
#         print(f"  Sequence {i} has {len(seq['x_gyro'])} windows")
#     print("\nValidation Sequences:")
#     for i, seq in enumerate(val_sequences):
#         print(f"  Sequence {i} has {len(seq['x_gyro'])} windows")

#     # Combine (stack) sequences to form single train & val datasets
#     train_dataset = concat_and_build_dataset(train_sequences)
#     val_dataset = concat_and_build_dataset(val_sequences)
#     print(f"\nTotal training samples: {len(train_dataset)}")
#     print(f"Total validation samples: {len(val_dataset)}\n")

#     # print general information ----------------------------------------------
#     print(f"window_size: {window_size}")
#     print(f"stride: {stride}")
#     print(f"batch_size: {batch_size}")
#     print(f"num_epochs: {num_epochs}")
#     print(f"learning_rate: {learning_rate}")
    
#     if scheduler_type == 'MultiStepLR':
#         print(f"lr_scheduler: {scheduler_type}, "
#               f"milestones: {scheduler_props['milestones']}, "
#               f"gamma: {scheduler_props['gamma']}")
#     else: # 'ReduceLROnPlateau'
#         print(f"lr_scheduler: {scheduler_type}, "
#               f"mode: {scheduler_props['mode']}, "
#               f"factor: {scheduler_props['factor']}, "
#               f"patience: {scheduler_props['patience']}, "
#               f"threshold: {scheduler_props['threshold']}, "
#               f"min_lr: {scheduler_props['min_lr']}")

#     # Create DataLoaders
#     train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, batch_size=batch_size)

#     # Create model and move to device
#     print(f"Creating model: {model_type}...")
#     model = create_model(window_size=window_size, model_type=model_type)
#     model.to(device)

#     # Create criterion, optimizer, and scheduler
#     criterion = CustomMultiLossLayer(nb_outputs=2)
#     optimizer = Adam(model.parameters(), lr=learning_rate)
    
#     if scheduler_props['type'] == 'MultiStepLR':
#         scheduler = MultiStepLR(optimizer,
#                                 milestones=scheduler_props['milestones'], 
#                                 gamma=scheduler_props['gamma'])
#     else: # 'ReduceLROnPlateau'
#         scheduler = ReduceLROnPlateau(optimizer,
#                                         mode=scheduler_props['mode'],
#                                         factor=scheduler_props['factor'],
#                                         patience=scheduler_props['patience'],
#                                         threshold=scheduler_props['threshold'],
#                                         min_lr=scheduler_props['min_lr'])

#     # -------------------------------------
#     # Build a hyperparams dict (we'll add a timestamp here)
#     from datetime import datetime
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     # Prepare checkpoint path
#     checkpoint_path = Path(f'{args.output}_{timestamp}.pt')
#     # Gather them all in one dictionary
#     hyperparams = { 
#         # General training
#         'model_type': model_type,
#         'sequence_length': window_size,
#         'batch_size': batch_size,
#         'learning_rate': learning_rate,
#         'lr_scheduler': lr_scheduler_props,
#         'optimizer': optimizer_name,
#         # For naming logs
#         'timestamp': timestamp
#     }
#     # Add model-specific hyperparameters
#     if model_type == 'lstm':
#         # for LSTM
#         hyperparams.update(LSTM_PROPERTIES)
#     elif model_type == 'tcn':
#         # for TCN
#         hyperparams.update(TCNBLOCK_PROPERTIES)

#     # -------------------------------------
#     # Train
#     metrics_dict = train_loop(
#         model=model,
#         train_loader=train_loader,
#         val_loader=val_loader,
#         criterion=criterion,
#         optimizer=optimizer,
#         device=device,
#         num_epochs=num_epochs,
#         scheduler=scheduler,
#         checkpoint_path=checkpoint_path,
#         hyperparams=hyperparams
#     )


# if __name__ == '__main__':
#     main()
