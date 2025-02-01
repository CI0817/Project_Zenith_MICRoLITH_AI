import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils import weight_norm

class QuaternionLoss:
    @staticmethod
    def quaternion_phi_3_error(y_true, y_pred):
        y_pred_normalized = F.normalize(y_pred, p=2, dim=-1)
        dot_product = torch.sum(y_true * y_pred_normalized, dim=-1)
        return torch.acos(torch.abs(dot_product))

    @staticmethod
    def quaternion_phi_4_error(y_true, y_pred):
        y_pred_normalized = F.normalize(y_pred, p=2, dim=-1)
        dot_product = torch.sum(y_true * y_pred_normalized, dim=-1)
        return 1 - torch.abs(dot_product)

    @staticmethod
    def quaternion_mean_multiplicative_error(y_true, y_pred):
        # Simplified quaternion multiplication error
        y_pred_normalized = F.normalize(y_pred, p=2, dim=-1)
        return F.mse_loss(y_true, y_pred_normalized)

class CustomMultiLossLayer(nn.Module):
    def __init__(self, nb_outputs=2):
        super().__init__()
        self.nb_outputs = nb_outputs
        self.log_vars = nn.Parameter(torch.zeros(nb_outputs))
        
    def forward(self, y_true_list, y_pred_list):
        loss = 0
        for i in range(self.nb_outputs):
            precision = torch.exp(-self.log_vars[i])
            if i == 0:
                loss += precision * F.l1_loss(y_pred_list[i], y_true_list[i]) + self.log_vars[i]
            else:
                loss += precision * QuaternionLoss.quaternion_mean_multiplicative_error(
                    y_true_list[i], y_pred_list[i]) + self.log_vars[i]
        return loss.mean()

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

TCNBLOCK_PROPERTIES = {
    'kernel_size': 7,
    'num_layers': 6,
    'dropout': 0.2,
    'activation': "gelu",
    'norm': "group",
}

class LSTM(nn.Module):
    def __init__(self, window_size=200):
        super().__init__()

        # LSTM properties/parameters
        lstm_props = LSTM_PROPERTIES
        
        # Convolutional layers for gyroscope
        self.convA1 = nn.Conv1d(3, 128, kernel_size=lstm_props['kernel_size'])
        self.convA2 = nn.Conv1d(128, 128, kernel_size=lstm_props['kernel_size'])
        
        # Convolutional layers for accelerometer
        self.convB1 = nn.Conv1d(3, 128, kernel_size=lstm_props['kernel_size'])
        self.convB2 = nn.Conv1d(128, 128, kernel_size=lstm_props['kernel_size'])
        
        # Pooling
        self.pool = nn.MaxPool1d(3)
        
        # Bidirectional LSTM layers
        lstm_hidden = lstm_props['hidden_size']
        self.lstm1 = nn.LSTM(256, lstm_hidden, bidirectional=lstm_props['bidirectional'], batch_first=lstm_props['batch_first'])
        self.lstm2 = nn.LSTM(lstm_hidden*2, lstm_hidden, bidirectional=lstm_props['bidirectional'], batch_first=lstm_props['batch_first'])
        
        # Dropout
        self.dropout = nn.Dropout(lstm_props['dropout'])
        
        # Output layers
        self.fc_pos = nn.Linear(lstm_hidden*2, 3)
        self.fc_quat = nn.Linear(lstm_hidden*2, 4)
    
        
    def forward(self, x_gyro, x_acc):
        # Transpose for conv1d which expects [batch, channels, length]
        x_gyro = x_gyro.transpose(1, 2)
        x_acc = x_acc.transpose(1, 2)
        
        # Process gyroscope data
        xa = F.relu(self.convA1(x_gyro))
        xa = F.relu(self.convA2(xa))
        xa = self.pool(xa)
        
        # Process accelerometer data
        xb = F.relu(self.convB1(x_acc))
        xb = F.relu(self.convB2(xb))
        xb = self.pool(xb)
        
        # Concatenate and transpose back
        x = torch.cat([xa, xb], dim=1).transpose(1, 2)
        
        # LSTM layers
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x[:, -1, :])  # Take last output
        
        # Output heads
        pos = self.fc_pos(x)
        quat = self.fc_quat(x)
        
        return pos, quat

def make_norm(num_channels, norm_type="group", num_groups=8):
    if norm_type == "group":
        return nn.GroupNorm(num_groups, num_channels)
    elif norm_type == "layer":
        return nn.LayerNorm(num_channels)
    else:
        return nn.BatchNorm1d(num_channels)  # fallback to batch norm

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1,
                 norm='group', activation=nn.GELU()):
        super().__init__()

        # TCN properties/parameters
        tcn_props = TCNBLOCK_PROPERTIES
        kernel_size = tcn_props['kernel_size']
        dropout = tcn_props['dropout']

        # activation function
        if tcn_props['activation'] == "gelu":
            activation = nn.GELU()
        elif tcn_props['activation'] == "relu":
            activation = nn.ReLU()

        padding = (kernel_size - 1) * dilation

        # Convolution block
        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels,
                                           kernel_size=kernel_size,
                                           stride=stride, padding=padding,
                                           dilation=dilation))
        self.norm1 = nn.GroupNorm(8, out_channels) if norm=='group' else nn.BatchNorm1d(out_channels)
        self.act1 = activation

        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels,
                                           kernel_size=kernel_size,
                                           stride=stride, padding=padding,
                                           dilation=dilation))
        self.norm2 = nn.GroupNorm(8, out_channels) if norm=='group' else nn.BatchNorm1d(out_channels)
        self.act2 = activation

        self.dropout = nn.Dropout(dropout)

        # Downsample if needed
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.GroupNorm(8, out_channels) if norm=='group' else nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        # First conv
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.dropout(out)

        # Second conv
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)
        out = self.dropout(out)

        # Residual
        res = x if self.downsample is None else self.downsample(x)

        # Match size if dimension mismatch
        if out.shape[-1] != res.shape[-1]:
            min_len = min(out.shape[-1], res.shape[-1])
            out = out[..., :min_len]
            res = res[..., :min_len]

        return out + res  # possibly wrap in activation: F.relu(out + res)

class TCNModel(nn.Module):
    def __init__(self, window_size=200):
        super().__init__()

        # TCN properties/parameters
        tcn_props = TCNBLOCK_PROPERTIES

        # activation function
        if tcn_props['activation'] == "gelu":
            activation = nn.GELU()
        elif tcn_props['activation'] == "relu":
            activation = nn.ReLU()

        # Deeper TCN for Gyro (6 layers)
        self.gyro_tcn = nn.Sequential(
            TCNBlock(3,   64, kernel_size=tcn_props['kernel_size'], dilation=1, norm=tcn_props['norm'] , activation=activation),
            TCNBlock(64, 128, kernel_size=tcn_props['kernel_size'], dilation=2, norm=tcn_props['norm'], activation=activation),
            TCNBlock(128,256, kernel_size=tcn_props['kernel_size'], dilation=4, norm=tcn_props['norm'], activation=activation),
            TCNBlock(256,256, kernel_size=tcn_props['kernel_size'], dilation=8, norm=tcn_props['norm'], activation=activation),
            TCNBlock(256,256, kernel_size=tcn_props['kernel_size'], dilation=16,norm=tcn_props['norm'], activation=activation),
            TCNBlock(256,256, kernel_size=tcn_props['kernel_size'], dilation=32,norm=tcn_props['norm'], activation=activation),
        )

        # Deeper TCN for Acc (6 layers)
        self.acc_tcn = nn.Sequential(
            TCNBlock(3,   64, kernel_size=tcn_props['kernel_size'], dilation=1, norm=tcn_props['norm'] , activation=activation),
            TCNBlock(64, 128, kernel_size=tcn_props['kernel_size'], dilation=2, norm=tcn_props['norm'], activation=activation),
            TCNBlock(128,256, kernel_size=tcn_props['kernel_size'], dilation=4, norm=tcn_props['norm'], activation=activation),
            TCNBlock(256,256, kernel_size=tcn_props['kernel_size'], dilation=8, norm=tcn_props['norm'], activation=activation),
            TCNBlock(256,256, kernel_size=tcn_props['kernel_size'], dilation=16,norm=tcn_props['norm'], activation=activation),
            TCNBlock(256,256, kernel_size=tcn_props['kernel_size'], dilation=32,norm=tcn_props['norm'], activation=activation),
        )

        # Combine features
        self.feature_processing = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=1),
            nn.GroupNorm(8, 512),
            activation,
            nn.Dropout(tcn_props['dropout']),
        )

        # Global pool
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Position / Quaternion heads
        self.pos_path = nn.Sequential(
            nn.Linear(512, 256),
            activation,
            nn.Dropout(tcn_props['dropout']),
            nn.Linear(256, 3),
        )
        self.quat_path = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 4),
        )

    def forward(self, x_gyro, x_acc):
        x_gyro = x_gyro.transpose(1, 2)  # B x C x T
        x_acc  = x_acc.transpose(1, 2)  # B x C x T

        x_gyro = self.gyro_tcn(x_gyro)
        x_acc  = self.acc_tcn(x_acc)

        # Concatenate: B x (256+256) x T
        x = torch.cat([x_gyro, x_acc], dim=1)

        x = self.feature_processing(x)
        x = self.global_pool(x).squeeze(-1)  # B x 512

        pos = self.pos_path(x)               # B x 3
        quat = self.quat_path(x)             # B x 4
        quat = F.normalize(quat, p=2, dim=-1)# normalize quaternion
        return pos, quat


# class TCNBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
#         super().__init__()

#         # TCN properties/parameters
#         tcn_props = TCNBLOCK_PROPERTIES

#         padding = (tcn_props['kernel_size'] - 1) * dilation
        
#         self.conv_block = nn.Sequential(
#             weight_norm(nn.Conv1d(in_channels, out_channels, tcn_props['kernel_size'],
#                                 stride=stride, padding=padding, dilation=dilation)),
#             nn.BatchNorm1d(out_channels),
#             nn.ReLU(),
#             nn.Dropout(tcn_props['dropout']),
#             weight_norm(nn.Conv1d(out_channels, out_channels, tcn_props['kernel_size'],
#                                 stride=stride, padding=padding, dilation=dilation)),
#             nn.BatchNorm1d(out_channels),
#             nn.ReLU(),
#             nn.Dropout(tcn_props['dropout'])
#         )
        
#         self.downsample = nn.Sequential(
#             nn.Conv1d(in_channels, out_channels, 1),
#             nn.BatchNorm1d(out_channels)
#         ) if in_channels != out_channels else None
        
#     def forward(self, x):
#         out = self.conv_block(x)
#         res = x if self.downsample is None else self.downsample(x)
        
#         # Match sizes if needed
#         if out.size(2) != res.size(2):
#             min_len = min(out.size(2), res.size(2))
#             out = out[:, :, :min_len]
#             res = res[:, :, :min_len]
            
#         return F.relu(out + res)

# class TCNModel(nn.Module):
#     def __init__(self, window_size=200):
#         super().__init__()

#         # TCN properties/parameters
#         tcn_props = TCNBLOCK_PROPERTIES
        
#         # Network capacity with 4 layers
#         self.gyro_tcn = nn.Sequential(
#             TCNBlock(3,     64,     kernel_size=7,  dilation=1),     # from 3 -> 64
#             TCNBlock(64,    128,    kernel_size=7,  dilation=2),
#             TCNBlock(128,   256,    kernel_size=7,  dilation=4),
#             TCNBlock(256,   256,    kernel_size=7,  dilation=8)
#         )
        
#         self.acc_tcn = nn.Sequential(
#             TCNBlock(3,     64,     kernel_size=7,  dilation=1),     # from 3 -> 64
#             TCNBlock(64,    128,    kernel_size=7,  dilation=2),
#             TCNBlock(128,   256,    kernel_size=7,  dilation=4),
#             TCNBlock(256,   256,    kernel_size=7,  dilation=8)
#         )
        
#         # Add feature processing layers
#         self.feature_processing = nn.Sequential(
#             nn.Conv1d(512, 512, kernel_size=1),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Dropout(tcn_props['dropout'])
#         )
        
#         # Global average pooling instead of flatten
#         self.global_pool = nn.AdaptiveAvgPool1d(1)
        
#         # Separate processing paths for position and quaternion
#         self.pos_path = nn.Sequential(
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Dropout(tcn_props['dropout']),
#             nn.Linear(256, 3)
#         )
        
#         self.quat_path = nn.Sequential(
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(256, 4)
#         )
        
#     def forward(self, x_gyro, x_acc):
#         # Process inputs
#         x_gyro = x_gyro.transpose(1, 2)
#         x_acc = x_acc.transpose(1, 2)
        
#         # TCN processing
#         x_gyro = self.gyro_tcn(x_gyro)
#         x_acc = self.acc_tcn(x_acc)
        
#         # Combine features
#         x = torch.cat([x_gyro, x_acc], dim=1)
#         x = self.feature_processing(x)
        
#         # Global pooling
#         x = self.global_pool(x).squeeze(-1)
        
#         # Generate outputs through separate paths
#         pos = self.pos_path(x)
#         quat = F.normalize(self.quat_path(x), p=2, dim=-1)  # Normalize quaternion
        
#         return pos, quat

class TransformerModel(nn.Module):
    def __init__(self, window_size=200, d_model=256, nhead=8, num_layers=3):
        super().__init__()
        
        # Initial projections
        self.gyro_proj = nn.Linear(3, d_model // 2)
        self.acc_proj = nn.Linear(3, d_model // 2)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, window_size, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=1024, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layers
        self.dropout = nn.Dropout(0.1)
        self.fc_pos = nn.Linear(d_model, 3)
        self.fc_quat = nn.Linear(d_model, 4)
        
    def forward(self, x_gyro, x_acc):
        # Project inputs
        x_gyro = self.gyro_proj(x_gyro)
        x_acc = self.acc_proj(x_acc)
        
        # Concatenate along feature dimension
        x = torch.cat([x_gyro, x_acc], dim=-1)
        
        # Add positional encoding
        x = x + self.pos_encoder
        
        # Transformer (transpose for sequence first)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)
        
        # Use the last sequence element for output
        x = x[:, -1]
        x = self.dropout(x)
        
        # Output heads
        pos = self.fc_pos(x)
        quat = self.fc_quat(x)
        
        return pos, quat

def create_model(window_size=200, model_type='lstm'):
    """Create a model of the specified type.
    
    Args:
        window_size: Size of the input window
        model_type: Type of model to create ('lstm', 'tcn', or 'transformer')
    """
    if model_type == 'lstm':
        return LSTM(window_size)
    elif model_type == 'tcn':
        return TCNModel(window_size)
    elif model_type == 'transformer':
        return TransformerModel(window_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}")