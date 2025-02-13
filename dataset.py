import numpy as np
import pandas as pd
import quaternion
import scipy.interpolate
import torch
from torch.utils.data import Dataset
import cv2

# === Helper Functions ===

def sliding_window_indices(total_length: int, window_size: int, stride: int):
    """
    Generate starting indices for sliding windows.
    """
    return range(0, total_length - window_size - 1, stride)

def stack_windows(windows: list) -> np.ndarray:
    """
    Stack a list of windows into a numpy array.
    """
    return np.array(windows)

def normalize_angle(angle: float) -> float:
    """
    Normalize an angle to the range [-pi, pi].
    """
    if angle < -np.pi:
        return angle + 2 * np.pi
    elif angle > np.pi:
        return angle - 2 * np.pi
    return angle

# === Dataset Class ===

class IMUDataset(Dataset):
    def __init__(self, gyro_data, acc_data, pos_data, ori_data, window_size: int = 200, stride: int = 10):
        """
        A PyTorch Dataset for IMU data that provides gyro, acc inputs and corresponding 
        delta position and delta quaternion outputs.
        """
        self.window_size = window_size
        self.stride = stride
        
        # Use the 6D dataset loader (with quaternion representation)
        [x_gyro, x_acc], [y_delta_p, y_delta_q], self.init_p, self.init_q = load_dataset_6d_quat(
            gyro_data, acc_data, pos_data, ori_data, window_size, stride
        )
        # Convert numpy arrays to PyTorch tensors
        self.x_gyro = torch.FloatTensor(x_gyro)
        self.x_acc = torch.FloatTensor(x_acc)
        self.y_delta_p = torch.FloatTensor(y_delta_p)
        self.y_delta_q = torch.FloatTensor(y_delta_q)
        
    def __len__(self):
        return len(self.x_gyro)
    
    def __getitem__(self, idx):
        return (self.x_gyro[idx], self.x_acc[idx]), (self.y_delta_p[idx], self.y_delta_q[idx])

# === Interpolation and Data Loading ===

def interpolate_3dvector_linear(data: np.ndarray, input_ts: np.ndarray, output_ts: np.ndarray) -> np.ndarray:
    """
    Linearly interpolate a 3D vector (e.g., gyro or acc data) from the input timestamps
    to the output timestamps.
    """
    assert data.shape[0] == input_ts.shape[0], "Data and timestamps must have the same length."
    interp_func = scipy.interpolate.interp1d(input_ts, data, axis=0)
    return interp_func(output_ts)

def load_euroc_mav_dataset(imu_file: str, gt_file: str):
    """
    Loads and interpolates the EuRoC MAV dataset.
    """
    gt_data = pd.read_csv(gt_file).values     # Format: timestamp, x, y, z, w, x, y, z
    imu_data = pd.read_csv(imu_file).values   # Format: timestamp, wx, wy, wz, ax, ay, az

    gyro_data = interpolate_3dvector_linear(imu_data[:, 1:4], imu_data[:, 0], gt_data[:, 0])
    acc_data = interpolate_3dvector_linear(imu_data[:, 4:7], imu_data[:, 0], gt_data[:, 0])
    
    pos_data = gt_data[:, 1:4]
    ori_data = gt_data[:, 4:8]

    return gyro_data, acc_data, pos_data, ori_data

def load_oxiod_dataset(imu_file: str, gt_file: str):
    """
    Loads the OXIOD dataset from CSV files and discards the first 1200 and last 300 samples.
    """
    imu_data = pd.read_csv(imu_file).values
    gt_data = pd.read_csv(gt_file).values

    imu_data = imu_data[1200:-300]
    gt_data = gt_data[1200:-300]

    gyro_data = imu_data[:, 4:7]
    acc_data = imu_data[:, 10:13]
    
    pos_data = gt_data[:, 2:5]
    # Orientation is stored as [w] and then [x, y, z]
    ori_data = np.concatenate([gt_data[:, 8:9], gt_data[:, 5:8]], axis=1)

    return gyro_data, acc_data, pos_data, ori_data

# === Quaternion and Coordinate Utilities ===

def force_quaternion_uniqueness(q: quaternion.quaternion) -> quaternion.quaternion:
    """
    Ensure a unique quaternion representation by flipping the sign if the first
    significant component is negative.
    """
    q_data = quaternion.as_float_array(q)
    for comp in q_data:
        if abs(comp) > 1e-5:
            return -q if comp < 0 else q
    return q  # In case all components are nearly zero

def cartesian_to_spherical_coordinates(point: np.ndarray):
    """
    Convert a 3D Cartesian point to spherical coordinates (delta_l, theta, psi),
    where delta_l is the radial distance, theta is the polar angle, and psi is the azimuth.
    """
    delta_l = np.linalg.norm(point)
    if delta_l > 1e-5:
        theta = np.arccos(point[2] / delta_l)
        psi = np.arctan2(point[1], point[0])
        return delta_l, theta, psi
    return 0.0, 0.0, 0.0

# === Dataset Loaders (6D, 3D, 2D) ===

def load_dataset_6d_rvec(imu_file: str, gt_file: str, window_size: int = 200, stride: int = 10):
    """
    Loads a 6D dataset where the orientation is represented using Rodrigues vectors.
    """
    imu_data = pd.read_csv(imu_file).values
    gt_data = pd.read_csv(gt_file).values

    # Concatenate gyro and acc data
    gyro_acc_data = np.concatenate([imu_data[:, 4:7], imu_data[:, 10:13]], axis=1)
    pos_data = gt_data[:, 2:5]
    ori_data = np.concatenate([gt_data[:, 8:9], gt_data[:, 5:8]], axis=1)

    # Use the midpoint for the initial reference
    mid_idx = window_size // 2 - stride // 2
    init_q = quaternion.from_float_array(ori_data[mid_idx, :])
    init_rmat = quaternion.as_rotation_matrix(init_q)
    init_rvec, _ = cv2.Rodrigues(init_rmat)
    init_tvec = pos_data[mid_idx, :]

    x_windows = []
    y_delta_rvec = []
    y_delta_tvec = []

    for idx in sliding_window_indices(gyro_acc_data.shape[0], window_size, stride):
        x_windows.append(gyro_acc_data[idx + 1 : idx + 1 + window_size, :])

        mid_a = idx + window_size // 2 - stride // 2
        mid_b = idx + window_size // 2 + stride // 2

        tvec_a = pos_data[mid_a, :]
        tvec_b = pos_data[mid_b, :]

        q_a = quaternion.from_float_array(ori_data[mid_a, :])
        q_b = quaternion.from_float_array(ori_data[mid_b, :])

        rmat_a = quaternion.as_rotation_matrix(q_a)
        rmat_b = quaternion.as_rotation_matrix(q_b)
        delta_rmat = rmat_b @ rmat_a.T

        delta_rvec, _ = cv2.Rodrigues(delta_rmat)
        # Compute delta position in the local coordinate frame of q_a
        delta_tvec = tvec_b - (delta_rmat @ tvec_a.T).T

        y_delta_rvec.append(delta_rvec)
        y_delta_tvec.append(delta_tvec)

    return (stack_windows(x_windows),
            [stack_windows(y_delta_rvec), stack_windows(y_delta_tvec)],
            init_rvec,
            init_tvec)

def load_dataset_6d_quat(gyro_data: np.ndarray, acc_data: np.ndarray, pos_data: np.ndarray, 
                           ori_data: np.ndarray, window_size: int = 200, stride: int = 10):
    """
    Loads a 6D dataset with orientation kept as quaternions.
    Returns: ([x_gyro, x_acc], [y_delta_p, y_delta_q], init_p, init_q)
    """
    if gyro_data.shape[0] < window_size:
        raise ValueError("Not enough data to create even a single window.")

    mid_idx = window_size // 2 - stride // 2
    init_p = pos_data[mid_idx, :]
    init_q = ori_data[mid_idx, :]

    x_gyro_windows = []
    x_acc_windows = []
    y_delta_p_windows = []
    y_delta_q_windows = []

    for idx in sliding_window_indices(gyro_data.shape[0], window_size, stride):
        x_gyro_windows.append(gyro_data[idx + 1 : idx + 1 + window_size, :])
        x_acc_windows.append(acc_data[idx + 1 : idx + 1 + window_size, :])

        p_a = pos_data[idx + window_size // 2 - stride // 2, :]
        p_b = pos_data[idx + window_size // 2 + stride // 2, :]

        q_a = quaternion.from_float_array(ori_data[idx + window_size // 2 - stride // 2, :])
        q_b = quaternion.from_float_array(ori_data[idx + window_size // 2 + stride // 2, :])

        # Transform position difference into q_a's local frame.
        delta_p = (quaternion.as_rotation_matrix(q_a).T @ (p_b - p_a).T).T
        delta_q = force_quaternion_uniqueness(q_a.conjugate() * q_b)

        y_delta_p_windows.append(delta_p)
        y_delta_q_windows.append(quaternion.as_float_array(delta_q))

    return ([stack_windows(x_gyro_windows), stack_windows(x_acc_windows)],
            [stack_windows(y_delta_p_windows), stack_windows(y_delta_q_windows)],
            init_p,
            init_q)

def load_dataset_3d(gyro_data: np.ndarray, acc_data: np.ndarray, loc_data: np.ndarray, 
                      window_size: int = 200, stride: int = 10):
    """
    Loads a 3D dataset for position tracking in spherical coordinates.
    Returns: ([x_gyro, x_acc], [y_delta_l, y_delta_theta, y_delta_psi], init_l, init_theta, init_psi)
    """
    ref_idx = window_size // 2 - stride // 2 - stride
    l0 = loc_data[ref_idx, :]
    l1 = loc_data[window_size // 2 - stride // 2, :]
    init_l = l1
    _, init_theta, init_psi = cartesian_to_spherical_coordinates(l1 - l0)

    x_gyro_windows = []
    x_acc_windows = []
    y_delta_l_windows = []
    y_delta_theta_windows = []
    y_delta_psi_windows = []

    for idx in sliding_window_indices(gyro_data.shape[0], window_size, stride):
        x_gyro_windows.append(gyro_data[idx + 1 : idx + 1 + window_size, :])
        x_acc_windows.append(acc_data[idx + 1 : idx + 1 + window_size, :])

        ref_mid = idx + window_size // 2 - stride // 2
        prev_ref = ref_mid - stride
        _, theta0, psi0 = cartesian_to_spherical_coordinates(loc_data[ref_mid, :] - loc_data[prev_ref, :])

        mid_a = idx + window_size // 2 - stride // 2
        mid_b = idx + window_size // 2 + stride // 2

        l_a = loc_data[mid_a, :]
        l_b = loc_data[mid_b, :]

        delta_l_val, theta1, psi1 = cartesian_to_spherical_coordinates(l_b - l_a)
        delta_theta = normalize_angle(theta1 - theta0)
        delta_psi = normalize_angle(psi1 - psi0)

        y_delta_l_windows.append(np.array([delta_l_val]))
        y_delta_theta_windows.append(np.array([delta_theta]))
        y_delta_psi_windows.append(np.array([delta_psi]))

    return ([stack_windows(x_gyro_windows), stack_windows(x_acc_windows)],
            [stack_windows(y_delta_l_windows), stack_windows(y_delta_theta_windows), stack_windows(y_delta_psi_windows)],
            init_l,
            init_theta,
            init_psi)

def load_dataset_2d(imu_file: str, gt_file: str, window_size: int = 200, stride: int = 10):
    """
    Loads a 2D dataset (x and y coordinates) for position tracking.
    Returns a windowed sequence and the corresponding delta (distance and heading).
    """
    imu_data = pd.read_csv(imu_file).values
    gt_data = pd.read_csv(gt_file).values

    gyro_acc_data = np.concatenate([imu_data[:, 4:7], imu_data[:, 10:13]], axis=1)
    loc_data = gt_data[:, 2:4]

    ref_idx = window_size // 2 - stride // 2 - stride
    l0 = loc_data[ref_idx, :]
    l1 = loc_data[window_size // 2 - stride // 2, :]
    init_l = l1
    psi0 = np.arctan2(l1[1] - l0[1], l1[0] - l0[0])
    init_psi = psi0

    x_windows = []
    y_delta_l_windows = []
    y_delta_psi_windows = []

    for idx in sliding_window_indices(gyro_acc_data.shape[0], window_size, stride):
        x_windows.append(gyro_acc_data[idx + 1 : idx + 1 + window_size, :])

        ref_mid = idx + window_size // 2 - stride // 2
        prev_ref = ref_mid - stride
        psi_prev = np.arctan2(
            loc_data[ref_mid, 1] - loc_data[prev_ref, 1],
            loc_data[ref_mid, 0] - loc_data[prev_ref, 0]
        )

        mid_a = idx + window_size // 2 - stride // 2
        mid_b = idx + window_size // 2 + stride // 2

        l_a = loc_data[mid_a, :]
        l_b = loc_data[mid_b, :]

        psi_curr = np.arctan2(l_b[1] - l_a[1], l_b[0] - l_a[0])
        delta_l_val = np.linalg.norm(l_b - l_a)
        delta_psi = normalize_angle(psi_curr - psi_prev)

        y_delta_l_windows.append(np.array([delta_l_val]))
        y_delta_psi_windows.append(np.array([delta_psi]))

    return (stack_windows(x_windows),
            [stack_windows(y_delta_l_windows), stack_windows(y_delta_psi_windows)],
            init_l,
            init_psi)


# import numpy as np
# import pandas as pd
# import quaternion
# import scipy.interpolate
# import torch
# from torch.utils.data import Dataset
# import cv2

# class IMUDataset(Dataset):
#     def __init__(self, gyro_data, acc_data, pos_data, ori_data, window_size=200, stride=10):
#         """
#         A PyTorch Dataset for IMU data, providing gyro, acc, and corresponding
#         delta position and orientation quaternions.
#         """
#         self.window_size = window_size  # Number of samples in each window
#         self.stride = stride            # Step size between consecutive windows
        
#         # Load and preprocess data using a custom 6D (pos + quat) function
#         [x_gyro, x_acc], [y_delta_p, y_delta_q], init_p, init_q = load_dataset_6d_quat(
#             gyro_data, acc_data, pos_data, ori_data, window_size, stride
#         )
        
#         # Convert numpy arrays to PyTorch tensors
#         self.x_gyro = torch.FloatTensor(x_gyro)       # Gyroscope data
#         self.x_acc = torch.FloatTensor(x_acc)         # Accelerometer data
#         self.y_delta_p = torch.FloatTensor(y_delta_p) # Delta position
#         self.y_delta_q = torch.FloatTensor(y_delta_q) # Delta quaternion
        
#     def __len__(self):
#         """
#         Returns the total number of samples in the dataset.
#         """
#         return len(self.x_gyro)
    
#     def __getitem__(self, idx):
#         """
#         Returns the gyro/acc inputs and the corresponding delta position/orientation.
#         """
#         return (self.x_gyro[idx], self.x_acc[idx]), (self.y_delta_p[idx], self.y_delta_q[idx])


# def interpolate_3dvector_linear(input, input_timestamp, output_timestamp):
#     """
#     Linearly interpolates a 3D vector (e.g., gyro/acc) from an input timestamp
#     array onto an output timestamp array.
#     """
#     assert input.shape[0] == input_timestamp.shape[0], \
#         "Input array and timestamps must have the same length."
    
#     # Create an interpolation function along axis=0 (the vector dimension)
#     func = scipy.interpolate.interp1d(input_timestamp, input, axis=0)
#     # Evaluate the interpolation at the new (output) timestamps
#     interpolated = func(output_timestamp)
#     return interpolated


# def load_euroc_mav_dataset(imu_data_filename, gt_data_filename):
#     """
#     Loads and interpolates EuRoC MAV dataset: 
#     - Reads IMU and ground truth data from CSV files.
#     - Interpolates the gyro/acc data to match ground truth timestamps.
#     """
#     gt_data = pd.read_csv(gt_data_filename).values     # Ground truth data (format: timestamp, x, y, z, w, x, y, z)
#     imu_data = pd.read_csv(imu_data_filename).values   # IMU data (format: timestamp, wx, wy, wz, ax, ay, az)

#     # Interpolate gyro and acc data to match GT timestamps
#     gyro_data = interpolate_3dvector_linear(imu_data[:, 1:4], imu_data[:, 0], gt_data[:, 0])
#     acc_data = interpolate_3dvector_linear(imu_data[:, 4:7], imu_data[:, 0], gt_data[:, 0])
    
#     # Positions and orientations come directly from GT data
#     pos_data = gt_data[:, 1:4]    # x, y, z
#     ori_data = gt_data[:, 4:8]    # w, x, y, z

#     return gyro_data, acc_data, pos_data, ori_data


# def load_oxiod_dataset(imu_data_filename, gt_data_filename):
#     """
#     Loads OXIOD dataset from CSV files. 
#     - Applies slicing to remove initial/final segments of data (potentially invalid).
#     - Extracts gyro, acc, pos, and quaternion orientation from columns.
#     """
#     imu_data = pd.read_csv(imu_data_filename).values  # IMU data 
#     gt_data = pd.read_csv(gt_data_filename).values    # Ground truth data

#     # Discard first 1200 samples and last 300 samples
#     imu_data = imu_data[1200:-300]
#     gt_data = gt_data[1200:-300]

#     # Extract gyro (columns 4:7) and acc (columns 10:13)
#     gyro_data = imu_data[:, 4:7]
#     acc_data = imu_data[:, 10:13]
    
#     # Extract pos (columns 2:5) and convert orientation (8:9 + 5:8)
#     pos_data = gt_data[:, 2:5]
#     # The orientation is stored as w (8) followed by x, y, z (5:8)
#     ori_data = np.concatenate([gt_data[:, 8:9], gt_data[:, 5:8]], axis=1)

#     return gyro_data, acc_data, pos_data, ori_data


# def force_quaternion_uniqueness(q):
#     """
#     Ensures a quaternion is uniquely represented by checking its first non-negligible
#     component and flipping sign if it is negative. This avoids ambiguity of quaternion sign.
#     """
#     q_data = quaternion.as_float_array(q)

#     # Check w, x, y, z in order, flip if the first significant component is negative
#     if np.absolute(q_data[0]) > 1e-05:
#         if q_data[0] < 0:
#             return -q
#         else:
#             return q
#     elif np.absolute(q_data[1]) > 1e-05:
#         if q_data[1] < 0:
#             return -q
#         else:
#             return q
#     elif np.absolute(q_data[2]) > 1e-05:
#         if q_data[2] < 0:
#             return -q
#         else:
#             return q
#     else:
#         if q_data[3] < 0:
#             return -q
#         else:
#             return q


# def cartesian_to_spherical_coordinates(point_cartesian):
#     """
#     Converts a 3D Cartesian point to spherical coordinates (delta_l, theta, psi),
#     where:
#         - delta_l is the radial distance
#         - theta is the polar angle from the z-axis
#         - psi is the azimuth angle in the xy-plane
#     """
#     delta_l = np.linalg.norm(point_cartesian)

#     # If the length is not negligible, compute angles
#     if np.absolute(delta_l) > 1e-05:
#         theta = np.arccos(point_cartesian[2] / delta_l)       # polar angle
#         psi = np.arctan2(point_cartesian[1], point_cartesian[0])  # azimuth
#         return delta_l, theta, psi
#     else:
#         # If near zero, return all zeros
#         return 0, 0, 0


# def load_dataset_6d_rvec(imu_data_filename, gt_data_filename, window_size=200, stride=10):
#     """
#     Loads a 6D dataset where orientation is represented via Rodrigues vectors (rvec).
#     - gyro_acc_data: concatenation of gyro and acc
#     - pos_data: x, y, z
#     - ori_data: w, x, y, z
#     - Returns (x, [y_delta_rvec, y_delta_tvec], init_rvec, init_tvec)
#     """
#     imu_data = pd.read_csv(imu_data_filename).values
#     gt_data = pd.read_csv(gt_data_filename).values

#     # Concatenate gyro (4:7) and acc (10:13)
#     gyro_acc_data = np.concatenate([imu_data[:, 4:7], imu_data[:, 10:13]], axis=1)
    
#     # Position data is columns 2:5; orientation is w + x, y, z
#     pos_data = gt_data[:, 2:5]
#     ori_data = np.concatenate([gt_data[:, 8:9], gt_data[:, 5:8]], axis=1)

#     # Extract initial quaternion around the midpoint
#     init_q = quaternion.from_float_array(ori_data[window_size//2 - stride//2, :])
    
#     # Convert init quaternion to rotation matrix and then to Rodrigues
#     init_rvec = np.empty((3, 1))
#     cv2.Rodrigues(quaternion.as_rotation_matrix(init_q), init_rvec)

#     init_tvec = pos_data[window_size//2 - stride//2, :]

#     x = []              # Will store sequences of gyro + acc
#     y_delta_rvec = []   # Delta rotation vectors
#     y_delta_tvec = []   # Delta translation vectors

#     # Slide over the dataset in steps of 'stride', collect windows of length 'window_size'
#     for idx in range(0, gyro_acc_data.shape[0] - window_size - 1, stride):
#         # Window of data from idx+1 to idx+1+window_size
#         x.append(gyro_acc_data[idx + 1 : idx + 1 + window_size, :])

#         # Positions at two instants
#         tvec_a = pos_data[idx + window_size//2 - stride//2, :]
#         tvec_b = pos_data[idx + window_size//2 + stride//2, :]

#         # Quaternions at two instants
#         q_a = quaternion.from_float_array(ori_data[idx + window_size//2 - stride//2, :])
#         q_b = quaternion.from_float_array(ori_data[idx + window_size//2 + stride//2, :])

#         # Convert quaternions to rotation matrices
#         rmat_a = quaternion.as_rotation_matrix(q_a)
#         rmat_b = quaternion.as_rotation_matrix(q_b)

#         # Compute relative rotation matrix
#         delta_rmat = np.matmul(rmat_b, rmat_a.T)

#         # Convert delta_rmat to Rodrigues vector
#         delta_rvec = np.empty((3, 1))
#         cv2.Rodrigues(delta_rmat, delta_rvec)

#         # Compute delta position in the local coordinate frame of q_a
#         delta_tvec = tvec_b - np.matmul(delta_rmat, tvec_a.T).T

#         y_delta_rvec.append(delta_rvec)
#         y_delta_tvec.append(delta_tvec)

#     # Reshape lists into numpy arrays
#     x = np.reshape(x, (len(x), x[0].shape[0], x[0].shape[1]))
#     y_delta_rvec = np.reshape(y_delta_rvec, (len(y_delta_rvec), y_delta_rvec[0].shape[0]))
#     y_delta_tvec = np.reshape(y_delta_tvec, (len(y_delta_tvec), y_delta_tvec[0].shape[0]))

#     return x, [y_delta_rvec, y_delta_tvec], init_rvec, init_tvec


# def load_dataset_6d_quat(gyro_data, acc_data, pos_data, ori_data, window_size=200, stride=10):
#     """
#     Similar to load_dataset_6d_rvec, except orientation is kept as quaternions (quat).
#     - gyro_data, acc_data, pos_data, ori_data are assumed to be already loaded (numpy arrays).
#     - Returns [x_gyro, x_acc], [y_delta_p, y_delta_q], init_p, init_q
#     """
#     if gyro_data.shape[0] < window_size:
#         raise ValueError("Not enough data to create even a single window.")

#     # The initial position and orientation are extracted around the midpoint
#     # The choice of choosign the midpoint is arbitrary
#     init_p = pos_data[window_size//2 - stride//2, :]
#     init_q = ori_data[window_size//2 - stride//2, :]

#     x_gyro = []
#     x_acc = []
#     y_delta_p = []
#     y_delta_q = []

#     # Slide over the dataset in steps of 'stride'
#     for idx in range(0, gyro_data.shape[0] - window_size - 1, stride):
#         # Collect gyro and acc windows
#         x_gyro.append(gyro_data[idx + 1 : idx + 1 + window_size, :])
#         x_acc.append(acc_data[idx + 1 : idx + 1 + window_size, :])

#         # Positions at two instants
#         p_a = pos_data[idx + window_size//2 - stride//2, :]
#         p_b = pos_data[idx + window_size//2 + stride//2, :]

#         # Quaternions at two instants
#         q_a = quaternion.from_float_array(ori_data[idx + window_size//2 - stride//2, :])
#         q_b = quaternion.from_float_array(ori_data[idx + window_size//2 + stride//2, :])

#         # Transform position difference into local frame of q_a
#         delta_p = np.matmul(quaternion.as_rotation_matrix(q_a).T, (p_b.T - p_a.T)).T

#         # Compute relative quaternion: q_a^(-1) * q_b, then ensure uniqueness
#         delta_q = force_quaternion_uniqueness(q_a.conjugate() * q_b)

#         y_delta_p.append(delta_p)
#         y_delta_q.append(quaternion.as_float_array(delta_q))

#     # Reshape everything into final numpy arrays
#     x_gyro = np.reshape(x_gyro, (len(x_gyro), x_gyro[0].shape[0], x_gyro[0].shape[1]))
#     x_acc = np.reshape(x_acc, (len(x_acc), x_acc[0].shape[0], x_acc[0].shape[1]))
#     y_delta_p = np.reshape(y_delta_p, (len(y_delta_p), y_delta_p[0].shape[0]))
#     y_delta_q = np.reshape(y_delta_q, (len(y_delta_q), y_delta_q[0].shape[0]))

#     return [x_gyro, x_acc], [y_delta_p, y_delta_q], init_p, init_q


# def load_dataset_3d(gyro_data, acc_data, loc_data, window_size=200, stride=10):
#     """
#     Loads a 3D dataset for position tracking in spherical coordinates.
#     - loc_data: x, y, z positions.
#     - We convert displacements in the local coordinate frame to spherical angles.
#     - Returns [x_gyro, x_acc], [y_delta_l, y_delta_theta, y_delta_psi], init_l, init_theta, init_psi
#     """
#     # Take an initial reference for position and angles
#     l0 = loc_data[window_size//2 - stride//2 - stride, :]
#     l1 = loc_data[window_size//2 - stride//2, :]
#     init_l = l1
#     # Convert to spherical to get initial angles
#     delta_l, init_theta, init_psi = cartesian_to_spherical_coordinates(l1 - l0)

#     x_gyro = []
#     x_acc = []
#     y_delta_l = []
#     y_delta_theta = []
#     y_delta_psi = []

#     # Slide over the dataset in steps of 'stride'
#     for idx in range(0, gyro_data.shape[0] - window_size - 1, stride):
#         # Collect gyro and acc windows
#         x_gyro.append(gyro_data[idx + 1 : idx + 1 + window_size, :])
#         x_acc.append(acc_data[idx + 1 : idx + 1 + window_size, :])

#         # Calculate spherical coords at the midpoint
#         delta_l0, theta0, psi0 = cartesian_to_spherical_coordinates(
#             loc_data[idx + window_size//2 - stride//2, :] -
#             loc_data[idx + window_size//2 - stride//2 - stride, :]
#         )

#         # Positions at two instants
#         l0 = loc_data[idx + window_size//2 - stride//2, :]
#         l1 = loc_data[idx + window_size//2 + stride//2, :]

#         # Convert displacement to spherical
#         delta_l, theta1, psi1 = cartesian_to_spherical_coordinates(l1 - l0)

#         # Differences in angles
#         delta_theta = theta1 - theta0
#         delta_psi = psi1 - psi0

#         # Ensure angles remain in [-π, π]
#         if delta_theta < -np.pi:
#             delta_theta += 2 * np.pi
#         elif delta_theta > np.pi:
#             delta_theta -= 2 * np.pi

#         if delta_psi < -np.pi:
#             delta_psi += 2 * np.pi
#         elif delta_psi > np.pi:
#             delta_psi -= 2 * np.pi

#         # Store the computed deltas
#         y_delta_l.append(np.array([delta_l]))
#         y_delta_theta.append(np.array([delta_theta]))
#         y_delta_psi.append(np.array([delta_psi]))

#     # Reshape into final numpy arrays
#     x_gyro = np.reshape(x_gyro, (len(x_gyro), x_gyro[0].shape[0], x_gyro[0].shape[1]))
#     x_acc = np.reshape(x_acc, (len(x_acc), x_acc[0].shape[0], x_acc[0].shape[1]))
#     y_delta_l = np.reshape(y_delta_l, (len(y_delta_l), y_delta_l[0].shape[0]))
#     y_delta_theta = np.reshape(y_delta_theta, (len(y_delta_theta), y_delta_theta[0].shape[0]))
#     y_delta_psi = np.reshape(y_delta_psi, (len(y_delta_psi), y_delta_psi[0].shape[0]))

#     return [x_gyro, x_acc], [y_delta_l, y_delta_theta, y_delta_psi], init_l, init_theta, init_psi


# def load_dataset_2d(imu_data_filename, gt_data_filename, window_size=200, stride=10):
#     """
#     Loads a 2D dataset (e.g., x and y coordinates only) for position tracking.
#     - gyro_acc_data: concatenation of gyro (4:7) and acc (10:13) from IMU file.
#     - loc_data: x, y from GT file.
#     - Returns a windowed sequence x of shape (N, window_size, 6) plus 
#       the deltas [y_delta_l, y_delta_psi] and initial (init_l, init_psi).
#     """
#     imu_data = pd.read_csv(imu_data_filename).values
#     gt_data = pd.read_csv(gt_data_filename).values
    
#     # Concatenate gyro (4:7) and acc (10:13)
#     gyro_acc_data = np.concatenate([imu_data[:, 4:7], imu_data[:, 10:13]], axis=1)
    
#     # Extract 2D location (x, y) from GT data
#     loc_data = gt_data[:, 2:4]

#     # Reference points to obtain initial heading
#     l0 = loc_data[window_size//2 - stride//2 - stride, :]
#     l1 = loc_data[window_size//2 - stride//2, :]

#     # Compute initial heading (psi0)
#     l_diff = l1 - l0
#     psi0 = np.arctan2(l_diff[1], l_diff[0])
#     init_l = l1
#     init_psi = psi0

#     x = []
#     y_delta_l = []
#     y_delta_psi = []

#     # Slide over the dataset in steps of 'stride'
#     for idx in range(0, gyro_acc_data.shape[0] - window_size - 1, stride):
#         # Collect a window of gyro+acc data
#         x.append(gyro_acc_data[idx + 1 : idx + 1 + window_size, :])

#         # Compute heading at the midpoint - stride
#         l0_diff = loc_data[idx + window_size//2 - stride//2, :] - \
#                   loc_data[idx + window_size//2 - stride//2 - stride, :]
#         psi0 = np.arctan2(l0_diff[1], l0_diff[0])

#         # Positions at two instants
#         l0 = loc_data[idx + window_size//2 - stride//2, :]
#         l1 = loc_data[idx + window_size//2 + stride//2, :]

#         # Compute displacement and heading
#         l_diff = l1 - l0
#         psi1 = np.arctan2(l_diff[1], l_diff[0])
#         delta_l = np.linalg.norm(l_diff)
#         delta_psi = psi1 - psi0

#         # Normalize heading to [-π, π]
#         if delta_psi < -np.pi:
#             delta_psi += 2 * np.pi
#         elif delta_psi > np.pi:
#             delta_psi -= 2 * np.pi

#         y_delta_l.append(np.array([delta_l]))
#         y_delta_psi.append(np.array([delta_psi]))

#     # Reshape into final numpy arrays
#     x = np.reshape(x, (len(x), x[0].shape[0], x[0].shape[1]))
#     y_delta_l = np.reshape(y_delta_l, (len(y_delta_l), y_delta_l[0].shape[0]))
#     y_delta_psi = np.reshape(y_delta_psi, (len(y_delta_psi), y_delta_psi[0].shape[0]))

#     return x, [y_delta_l, y_delta_psi], init_l, init_psi
