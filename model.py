import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

# === Loss Functions ===

class QuaternionLoss:
    @staticmethod
    def _normalized_dot(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Normalize y_pred and compute dot product with y_true."""
        y_pred_norm = F.normalize(y_pred, p=2, dim=-1)
        return torch.sum(y_true * y_pred_norm, dim=-1)

    @staticmethod
    def quaternion_phi_3_error(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        dot_product = QuaternionLoss._normalized_dot(y_true, y_pred)
        # Clamp to [0, 1] to avoid numerical issues
        return torch.acos(torch.clamp(torch.abs(dot_product), 0, 1))

    @staticmethod
    def quaternion_phi_4_error(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        dot_product = QuaternionLoss._normalized_dot(y_true, y_pred)
        return 1 - torch.abs(dot_product)

    @staticmethod
    def quaternion_mean_multiplicative_error(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        y_pred_norm = F.normalize(y_pred, p=2, dim=-1)
        return F.mse_loss(y_true, y_pred_norm)

class CustomMultiLossLayer(nn.Module):
    def __init__(self, nb_outputs: int = 2) -> None:
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(nb_outputs))
        
    def forward(self, y_true_list: list, y_pred_list: list) -> torch.Tensor:
        loss = 0.0
        for i, (y_true, y_pred) in enumerate(zip(y_true_list, y_pred_list)):
            precision = torch.exp(-self.log_vars[i])
            if i == 0:
                loss += precision * F.l1_loss(y_pred, y_true) + self.log_vars[i]
            else:
                loss += precision * QuaternionLoss.quaternion_mean_multiplicative_error(y_true, y_pred) + self.log_vars[i]
        return loss.mean()

# === Helper Functions ===

def make_norm(num_channels: int, norm_type: str = "group", num_groups: int = 8) -> nn.Module:
    """Return a normalization layer based on the given type."""
    if norm_type == "group":
        return nn.GroupNorm(num_groups, num_channels)
    elif norm_type == "layer":
        return nn.LayerNorm(num_channels)
    else:
        return nn.BatchNorm1d(num_channels)

# === LSTM Model ===

LSTM_PROPERTIES = {
    'input_size': 6,        # 3 gyroscope + 3 accelerometer
    'hidden_size': 128,
    'dropout': 0.5,
    'kernel_size': 11,
    'num_layers': 2,
    'batch_first': True,
    'bidirectional': True,
    'output_size': 7        # 3 position + 4 quaternion
}

def make_sensor_conv(kernel_size: int) -> nn.Sequential:
    """Return a sensor-specific conv block (for gyroscope or accelerometer)."""
    return nn.Sequential(
        nn.Conv1d(3, 128, kernel_size=kernel_size),
        nn.ReLU(),
        nn.Conv1d(128, 128, kernel_size=kernel_size),
        nn.ReLU(),
        nn.MaxPool1d(3)
    )

class LSTMModel(nn.Module):
    def __init__(self, window_size: int = 200) -> None:
        super().__init__()
        props = LSTM_PROPERTIES
        
        # Define separate convolutional blocks for each sensor.
        self.gyro_conv = make_sensor_conv(props['kernel_size'])
        self.acc_conv  = make_sensor_conv(props['kernel_size'])
        
        # LSTM layers (bidirectional)
        lstm_hidden = props['hidden_size']
        self.lstm1 = nn.LSTM(256, lstm_hidden, bidirectional=props['bidirectional'],
                             batch_first=props['batch_first'])
        self.lstm2 = nn.LSTM(lstm_hidden * 2, lstm_hidden, bidirectional=props['bidirectional'],
                             batch_first=props['batch_first'])
        
        self.dropout = nn.Dropout(props['dropout'])
        
        # Output heads for position and quaternion.
        self.fc_pos  = nn.Linear(lstm_hidden * 2, 3)
        self.fc_quat = nn.Linear(lstm_hidden * 2, 4)
        
    def forward(self, x_gyro: torch.Tensor, x_acc: torch.Tensor) -> tuple:
        # Conv1d expects shape [B, C, T]
        x_gyro = self.gyro_conv(x_gyro.transpose(1, 2))
        x_acc  = self.acc_conv(x_acc.transpose(1, 2))
        
        # Concatenate along channel dimension and transpose back to [B, T, C]
        x = torch.cat([x_gyro, x_acc], dim=1).transpose(1, 2)
        
        # Process with LSTM layers and take the final time step.
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x[:, -1, :])
        
        pos  = self.fc_pos(x)
        quat = self.fc_quat(x)
        return pos, quat

# === TCN Block and Model ===

TCNBLOCK_PROPERTIES = {
    'kernel_size': 7,
    'num_layers': 6,
    'dropout': 0.2,
    'activation': "gelu",
    'norm': "group",
}

class TCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, dilation: int = 1, norm: str = 'group',
                 activation: nn.Module = None, dropout: float = 0.2) -> None:
        super().__init__()
        if activation is None:
            activation = nn.GELU()
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels,
                                           kernel_size=kernel_size, stride=stride,
                                           padding=padding, dilation=dilation))
        self.norm1 = make_norm(out_channels, norm_type=norm)
        self.act1  = activation
        
        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels,
                                           kernel_size=kernel_size, stride=stride,
                                           padding=padding, dilation=dilation))
        self.norm2 = make_norm(out_channels, norm_type=norm)
        self.act2  = activation
        
        self.dropout = nn.Dropout(dropout)
        
        # Downsample the residual if needed.
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                make_norm(out_channels, norm_type=norm)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)
        out = self.dropout(out)
        
        res = x if self.downsample is None else self.downsample(x)
        # Adjust temporal dimensions if necessary.
        if out.shape[-1] != res.shape[-1]:
            min_len = min(out.shape[-1], res.shape[-1])
            out = out[..., :min_len]
            res = res[..., :min_len]
        return out + res

def build_tcn_stack(in_channels: int, channel_sizes: list, kernel_size: int,
                    norm: str, activation: nn.Module, dropout: float) -> nn.Sequential:
    """Build a stack of TCNBlocks with exponentially increasing dilation."""
    layers = []
    dilation = 1
    current_channels = in_channels
    for out_channels in channel_sizes:
        layers.append(
            TCNBlock(current_channels, out_channels, kernel_size=kernel_size,
                     dilation=dilation, norm=norm, activation=activation, dropout=dropout)
        )
        current_channels = out_channels
        dilation *= 2
    return nn.Sequential(*layers)

class TCNModel(nn.Module):
    def __init__(self, window_size: int = 200) -> None:
        super().__init__()
        props = TCNBLOCK_PROPERTIES
        activation = nn.GELU() if props['activation'] == "gelu" else nn.ReLU()
        
        # Define the channel progression for both sensor streams.
        channel_sizes = [64, 128, 256, 256, 256, 256]
        self.gyro_tcn = build_tcn_stack(in_channels=3, channel_sizes=channel_sizes,
                                        kernel_size=props['kernel_size'], norm=props['norm'],
                                        activation=activation, dropout=props['dropout'])
        self.acc_tcn = build_tcn_stack(in_channels=3, channel_sizes=channel_sizes,
                                       kernel_size=props['kernel_size'], norm=props['norm'],
                                       activation=activation, dropout=props['dropout'])
        
        # Combine and process concatenated features.
        self.feature_processing = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=1),
            make_norm(512, norm_type=props['norm']),
            activation,
            nn.Dropout(props['dropout']),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Separate output heads.
        self.pos_path = nn.Sequential(
            nn.Linear(512, 256),
            activation,
            nn.Dropout(props['dropout']),
            nn.Linear(256, 3),
        )
        self.quat_path = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 4),
        )
    
    def forward(self, x_gyro: torch.Tensor, x_acc: torch.Tensor) -> tuple:
        # Convert to [B, C, T] for TCNs.
        x_gyro = x_gyro.transpose(1, 2)
        x_acc  = x_acc.transpose(1, 2)
        
        x_gyro = self.gyro_tcn(x_gyro)
        x_acc  = self.acc_tcn(x_acc)
        # Concatenate along the channel dimension.
        x = torch.cat([x_gyro, x_acc], dim=1)
        
        x = self.feature_processing(x)
        x = self.global_pool(x).squeeze(-1)
        
        pos = self.pos_path(x)
        quat = F.normalize(self.quat_path(x), p=2, dim=-1)
        return pos, quat

def create_model(window_size: int = 200, model_type: str = 'lstm') -> nn.Module:
    """
    Create and return a model of the specified type.
    
    Args:
        window_size: Size of the input window.
        model_type: 'lstm' or 'tcn'.
    """
    if model_type.lower() == 'lstm':
        return LSTMModel(window_size)
    elif model_type.lower() == 'tcn':
        return TCNModel(window_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.utils.rnn import PackedSequence
# from torch.nn.utils import weight_norm

# class QuaternionLoss:
#     @staticmethod
#     def quaternion_phi_3_error(y_true, y_pred):
#         y_pred_normalized = F.normalize(y_pred, p=2, dim=-1)
#         dot_product = torch.sum(y_true * y_pred_normalized, dim=-1)
#         return torch.acos(torch.abs(dot_product))

#     @staticmethod
#     def quaternion_phi_4_error(y_true, y_pred):
#         y_pred_normalized = F.normalize(y_pred, p=2, dim=-1)
#         dot_product = torch.sum(y_true * y_pred_normalized, dim=-1)
#         return 1 - torch.abs(dot_product)

#     @staticmethod
#     def quaternion_mean_multiplicative_error(y_true, y_pred):
#         # Simplified quaternion multiplication error
#         y_pred_normalized = F.normalize(y_pred, p=2, dim=-1)
#         return F.mse_loss(y_true, y_pred_normalized)

# class CustomMultiLossLayer(nn.Module):
#     def __init__(self, nb_outputs=2):
#         super().__init__()
#         self.nb_outputs = nb_outputs
#         self.log_vars = nn.Parameter(torch.zeros(nb_outputs))
        
#     def forward(self, y_true_list, y_pred_list):
#         loss = 0
#         for i in range(self.nb_outputs):
#             precision = torch.exp(-self.log_vars[i])
#             if i == 0:
#                 loss += precision * F.l1_loss(y_pred_list[i], y_true_list[i]) + self.log_vars[i]
#             else:
#                 loss += precision * QuaternionLoss.quaternion_mean_multiplicative_error(
#                     y_true_list[i], y_pred_list[i]) + self.log_vars[i]
#         return loss.mean()

# LSTM_PROPERTIES = {
#     'input_size': 6,        # 3 gyroscope + 3 accelerometer
#     'hidden_size': 128,
#     'dropout': 0.5,
#     'kernel_size': 11,
#     'num_layers': 2,
#     'batch_first': True,
#     'bidirectional': True,
#     'output_size': 7        # 3 position + 4 quaternion
# }

# TCNBLOCK_PROPERTIES = {
#     'kernel_size': 7,
#     'num_layers': 6,
#     'dropout': 0.2,
#     'activation': "gelu",
#     'norm': "group",
# }

# class LSTM(nn.Module):
#     def __init__(self, window_size=200):
#         super().__init__()

#         # LSTM properties/parameters
#         lstm_props = LSTM_PROPERTIES
        
#         # Convolutional layers for gyroscope
#         self.convA1 = nn.Conv1d(3, 128, kernel_size=lstm_props['kernel_size'])
#         self.convA2 = nn.Conv1d(128, 128, kernel_size=lstm_props['kernel_size'])
        
#         # Convolutional layers for accelerometer
#         self.convB1 = nn.Conv1d(3, 128, kernel_size=lstm_props['kernel_size'])
#         self.convB2 = nn.Conv1d(128, 128, kernel_size=lstm_props['kernel_size'])
        
#         # Pooling
#         self.pool = nn.MaxPool1d(3)
        
#         # Bidirectional LSTM layers
#         lstm_hidden = lstm_props['hidden_size']
#         self.lstm1 = nn.LSTM(256, lstm_hidden, bidirectional=lstm_props['bidirectional'], batch_first=lstm_props['batch_first'])
#         self.lstm2 = nn.LSTM(lstm_hidden*2, lstm_hidden, bidirectional=lstm_props['bidirectional'], batch_first=lstm_props['batch_first'])
        
#         # Dropout
#         self.dropout = nn.Dropout(lstm_props['dropout'])
        
#         # Output layers
#         self.fc_pos = nn.Linear(lstm_hidden*2, 3)
#         self.fc_quat = nn.Linear(lstm_hidden*2, 4)
    
        
#     def forward(self, x_gyro, x_acc):
#         # Transpose for conv1d which expects [batch, channels, length]
#         x_gyro = x_gyro.transpose(1, 2)
#         x_acc = x_acc.transpose(1, 2)
        
#         # Process gyroscope data
#         xa = F.relu(self.convA1(x_gyro))
#         xa = F.relu(self.convA2(xa))
#         xa = self.pool(xa)
        
#         # Process accelerometer data
#         xb = F.relu(self.convB1(x_acc))
#         xb = F.relu(self.convB2(xb))
#         xb = self.pool(xb)
        
#         # Concatenate and transpose back
#         x = torch.cat([xa, xb], dim=1).transpose(1, 2)
        
#         # LSTM layers
#         x, _ = self.lstm1(x)
#         x = self.dropout(x)
#         x, _ = self.lstm2(x)
#         x = self.dropout(x[:, -1, :])  # Take last output
        
#         # Output heads
#         pos = self.fc_pos(x)
#         quat = self.fc_quat(x)
        
#         return pos, quat

# def make_norm(num_channels, norm_type="group", num_groups=8):
#     if norm_type == "group":
#         return nn.GroupNorm(num_groups, num_channels)
#     elif norm_type == "layer":
#         return nn.LayerNorm(num_channels)
#     else:
#         return nn.BatchNorm1d(num_channels)  # fallback to batch norm

# class TCNBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1,
#                  norm='group', activation=nn.GELU()):
#         super().__init__()

#         # TCN properties/parameters
#         tcn_props = TCNBLOCK_PROPERTIES
#         kernel_size = tcn_props['kernel_size']
#         dropout = tcn_props['dropout']

#         # activation function
#         if tcn_props['activation'] == "gelu":
#             activation = nn.GELU()
#         elif tcn_props['activation'] == "relu":
#             activation = nn.ReLU()

#         padding = (kernel_size - 1) * dilation

#         # Convolution block
#         self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels,
#                                            kernel_size=kernel_size,
#                                            stride=stride, padding=padding,
#                                            dilation=dilation))
#         self.norm1 = nn.GroupNorm(8, out_channels) if norm=='group' else nn.BatchNorm1d(out_channels)
#         self.act1 = activation

#         self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels,
#                                            kernel_size=kernel_size,
#                                            stride=stride, padding=padding,
#                                            dilation=dilation))
#         self.norm2 = nn.GroupNorm(8, out_channels) if norm=='group' else nn.BatchNorm1d(out_channels)
#         self.act2 = activation

#         self.dropout = nn.Dropout(dropout)

#         # Downsample if needed
#         self.downsample = None
#         if in_channels != out_channels:
#             self.downsample = nn.Sequential(
#                 nn.Conv1d(in_channels, out_channels, 1),
#                 nn.GroupNorm(8, out_channels) if norm=='group' else nn.BatchNorm1d(out_channels)
#             )

#     def forward(self, x):
#         # First conv
#         out = self.conv1(x)
#         out = self.norm1(out)
#         out = self.act1(out)
#         out = self.dropout(out)

#         # Second conv
#         out = self.conv2(out)
#         out = self.norm2(out)
#         out = self.act2(out)
#         out = self.dropout(out)

#         # Residual
#         res = x if self.downsample is None else self.downsample(x)

#         # Match size if dimension mismatch
#         if out.shape[-1] != res.shape[-1]:
#             min_len = min(out.shape[-1], res.shape[-1])
#             out = out[..., :min_len]
#             res = res[..., :min_len]

#         return out + res  # possibly wrap in activation: F.relu(out + res)

# class TCNModel(nn.Module):
#     def __init__(self, window_size=200):
#         super().__init__()

#         # TCN properties/parameters
#         tcn_props = TCNBLOCK_PROPERTIES

#         # activation function
#         if tcn_props['activation'] == "gelu":
#             activation = nn.GELU()
#         elif tcn_props['activation'] == "relu":
#             activation = nn.ReLU()

#         # Deeper TCN for Gyro (6 layers)
#         self.gyro_tcn = nn.Sequential(
#             TCNBlock(3,   64, kernel_size=tcn_props['kernel_size'], dilation=1, norm=tcn_props['norm'] , activation=activation),
#             TCNBlock(64, 128, kernel_size=tcn_props['kernel_size'], dilation=2, norm=tcn_props['norm'], activation=activation),
#             TCNBlock(128,256, kernel_size=tcn_props['kernel_size'], dilation=4, norm=tcn_props['norm'], activation=activation),
#             TCNBlock(256,256, kernel_size=tcn_props['kernel_size'], dilation=8, norm=tcn_props['norm'], activation=activation),
#             TCNBlock(256,256, kernel_size=tcn_props['kernel_size'], dilation=16,norm=tcn_props['norm'], activation=activation),
#             TCNBlock(256,256, kernel_size=tcn_props['kernel_size'], dilation=32,norm=tcn_props['norm'], activation=activation),
#         )

#         # Deeper TCN for Acc (6 layers)
#         self.acc_tcn = nn.Sequential(
#             TCNBlock(3,   64, kernel_size=tcn_props['kernel_size'], dilation=1, norm=tcn_props['norm'] , activation=activation),
#             TCNBlock(64, 128, kernel_size=tcn_props['kernel_size'], dilation=2, norm=tcn_props['norm'], activation=activation),
#             TCNBlock(128,256, kernel_size=tcn_props['kernel_size'], dilation=4, norm=tcn_props['norm'], activation=activation),
#             TCNBlock(256,256, kernel_size=tcn_props['kernel_size'], dilation=8, norm=tcn_props['norm'], activation=activation),
#             TCNBlock(256,256, kernel_size=tcn_props['kernel_size'], dilation=16,norm=tcn_props['norm'], activation=activation),
#             TCNBlock(256,256, kernel_size=tcn_props['kernel_size'], dilation=32,norm=tcn_props['norm'], activation=activation),
#         )

#         # Combine features
#         self.feature_processing = nn.Sequential(
#             nn.Conv1d(512, 512, kernel_size=1),
#             nn.GroupNorm(8, 512),
#             activation,
#             nn.Dropout(tcn_props['dropout']),
#         )

#         # Global pool
#         self.global_pool = nn.AdaptiveAvgPool1d(1)

#         # Position / Quaternion heads
#         self.pos_path = nn.Sequential(
#             nn.Linear(512, 256),
#             activation,
#             nn.Dropout(tcn_props['dropout']),
#             nn.Linear(256, 3),
#         )
#         self.quat_path = nn.Sequential(
#             nn.Linear(512, 256),
#             nn.GELU(),
#             nn.Dropout(0.2),
#             nn.Linear(256, 4),
#         )

#     def forward(self, x_gyro, x_acc):
#         x_gyro = x_gyro.transpose(1, 2)  # B x C x T
#         x_acc  = x_acc.transpose(1, 2)  # B x C x T

#         x_gyro = self.gyro_tcn(x_gyro)
#         x_acc  = self.acc_tcn(x_acc)

#         # Concatenate: B x (256+256) x T
#         x = torch.cat([x_gyro, x_acc], dim=1)

#         x = self.feature_processing(x)
#         x = self.global_pool(x).squeeze(-1)  # B x 512

#         pos = self.pos_path(x)               # B x 3
#         quat = self.quat_path(x)             # B x 4
#         quat = F.normalize(quat, p=2, dim=-1)# normalize quaternion
#         return pos, quat

# def create_model(window_size=200, model_type='lstm'):
#     """Create a model of the specified type.
    
#     Args:
#         window_size: Size of the input window
#         model_type: Type of model to create ('lstm', 'tcn')
#     """
#     if model_type == 'lstm':
#         return LSTM(window_size)
#     elif model_type == 'tcn':
#         return TCNModel(window_size)
#     else:
#         raise ValueError(f"Unknown model type: {model_type}")