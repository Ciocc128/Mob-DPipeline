# Import necessary libraries for data manipulation, signal processing, and rotations
# %%
from mobgap.data import GenericMobilisedDataset
import numpy as np
from scipy.spatial.transform import Rotation as R
from gaitmap.utils.rotations import rotate_dataset
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import os
#TODO remove the reorientation matrix from SMP framework to MOBGAP framwork and mantain anly the part of alligning to the gravity vector.
# --- Utility Functions ---

def unit(vector):
    """
    Normalize a vector to unit length.

    Parameters:
        vector (np.ndarray): Input vector to normalize.

    Returns:
        np.ndarray: Normalized unit vector.
    """
    return vector / np.linalg.norm(vector)


def compute_envelope(signal):
    """
    Compute the envelope of a signal using the Hilbert transform.

    Parameters:
        signal (np.ndarray): The input signal (e.g., magnitude of gyroscope data).

    Returns:
        np.ndarray: The envelope of the signal.
    """
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    return envelope

def moving_average(signal, window_size=100):
    """
    Compute the moving average of a signal.

    Parameters:
        signal (np.ndarray): Input signal for which to calculate the moving average.
        window_size (int): The size of the window over which to compute the moving average.

    Returns:
        np.ndarray: The moving average of the input signal.
    """
    return np.convolve(signal, np.ones(window_size) / window_size, mode='same')

def detect_stationary_period(gyr_data, threshold=0.6, window_size=300, visualize=False, ma_window=200):
    """
    Detect the longest stationary period in gyroscope data based on the signal's envelope and moving average.

    Parameters:
        gyr_data (np.ndarray): An N x 3 array of gyroscope data.
        threshold (float): Threshold for the envelope to determine the stationary state.
        window_size (int): Size of the window for calculating variance in the stationary state.
        visualize (bool): Whether to visualize the envelope and the stationary detection (default is False).
        ma_window (int): Window size for the moving average.

    Returns:
        np.ndarray: Subset of data during the detected stationary period.
    """
    # Calculate the magnitude of the gyroscope data (sqrt(gyr_x^2 + gyr_y^2 + gyr_z^2))
    gyr_magnitude = np.linalg.norm(gyr_data, axis=1)

    # Compute the envelope of the magnitude signal
    envelope = compute_envelope(gyr_magnitude)
    envelopeNoOffset = envelope - np.mean(envelope)

    # Compute the moving average of the envelope
    envelope_ma = moving_average(abs(envelopeNoOffset), window_size=ma_window)

    # Identify stationary regions where the envelope is below the threshold
    stationary_mask = envelope_ma < threshold

    # Find the longest consecutive sequence of True values in the stationary mask
    longest_seq_start = None
    longest_seq_length = 0

    current_seq_start = None
    current_seq_length = 0

    for i, is_stationary in enumerate(stationary_mask):
        if is_stationary:
            if current_seq_start is None:
                current_seq_start = i  # Start of a new stationary sequence
            current_seq_length += 1
        else:
            if current_seq_length > longest_seq_length:
                longest_seq_start = current_seq_start
                longest_seq_length = current_seq_length
            current_seq_start = None
            current_seq_length = 0

    # Handle case where the longest sequence ends at the last sample
    if current_seq_length > longest_seq_length:
        longest_seq_start = current_seq_start
        longest_seq_length = current_seq_length

    if longest_seq_start is None:
        print("No clear stationary period detected. Defaulting to first part of the signal.")
        # if no stationary period is detected, return the first 3s of the signal
        return gyr_data[:window_size]

    # Get the longest stationary data
    start_idx = longest_seq_start
    end_idx = start_idx + longest_seq_length
    stationary_data = gyr_data[start_idx:end_idx]

    # Visualize the envelope and detected stationary period if needed
    if visualize:
        plt.figure(figsize=(12, 6))
        plt.plot(envelopeNoOffset, label="Envelope without offset")
        plt.plot(envelope_ma, label="Moving Average", color='orange')  # Add the moving average line
        plt.axhline(y=threshold, color='r', linestyle='--', label="Threshold")
        plt.axvspan(start_idx, end_idx, color='green', alpha=0.3, label="Detected Stationary Period")
        plt.title("Envelope of Gyroscope Data Magnitude with Detected Stationary Period")
        plt.xlabel("Sample")
        plt.ylabel("Envelope Value")
        plt.legend()
        plt.show()

    # Debug: print some indices
    print(f"Start index of the longest stationary period: {start_idx}")
    print(f"End index of the longest stationary period: {end_idx}")

    return stationary_data



def calc_R(acc_data):
    """
    Calculate the rotation matrix to align the accelerometer data with the gravity vector.

    Parameters:
        acc_data (np.ndarray): An N x 3 array of accelerometer data.

    Returns:
        np.ndarray: The rotation matrix (3x3) to align the data with the gravity vector.
    """
    # Ideal gravity vector, assumed to be along the X-axis
    gravity_local_ideal = np.array([1, 0, 0])

    # Detect the stationary period and compute the real gravity vector
    stationary_data = detect_stationary_period(acc_data, visualize=False) # visualize=True if you want to run this script
    gravity_local_real = unit(np.mean(stationary_data, axis=0))

    # Log the detected gravity vector for verification
    print(f"Real gravity vector: {gravity_local_real}")

    # Calculate the angle between the real and ideal gravity vectors
    angle = np.degrees(np.arccos(np.clip(np.dot(gravity_local_real, gravity_local_ideal), -1.0, 1.0)))

    # Calculate the rotation axis using the cross product
    rotation_axis = unit(np.cross(gravity_local_real, gravity_local_ideal))
    half_angle_rad = np.radians(angle / 2)

    # Create a quaternion from the angle and axis, and convert it to a rotation matrix
    q = np.hstack(([np.cos(half_angle_rad)], rotation_axis * np.sin(half_angle_rad)))
    rotation_matrix = R.from_quat(q).as_matrix()

    return rotation_matrix


def rotate_with_gaitmap(data, rotation_matrix):
    """
    Rotate the accelerometer and gyroscope data using a rotation matrix.

    Parameters:
        data (pd.DataFrame): DataFrame containing accelerometer and gyroscope data.
        rotation_matrix (np.ndarray): A 3x3 rotation matrix.

    Returns:
        pd.DataFrame: The rotated dataset.
    """
    rotation = R.from_matrix(rotation_matrix)

    required_columns = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
    if not all(col in data.columns for col in required_columns):
        raise ValueError("DataFrame must contain 'acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z' columns")

    # Rotate the accelerometer and gyroscope data
    rotated_data = rotate_dataset(data, rotation)

    # Update the DataFrame with the rotated data
    data = rotated_data

    return data

def plot_data(original_data, aligned_data, title):
    """
    Plot the original and aligned accelerometer data for comparison.

    Parameters:
        original_data (pd.DataFrame): Original accelerometer data (before rotation).
        aligned_data (pd.DataFrame): Aligned accelerometer data (after rotation).
        title (str): Title for the plot.
    """
    plt.figure(figsize=(12, 6))

    # Original data plot
    plt.subplot(1, 2, 1)
    plt.plot(original_data.index, original_data['acc_x'], label='acc_x (Original)', color='r')
    plt.plot(original_data.index, original_data['acc_y'], label='acc_y (Original)', color='g')
    plt.plot(original_data.index, original_data['acc_z'], label='acc_z (Original)', color='b')
    plt.title(f'{title} - Original Data')
    plt.xlabel('Time')
    plt.ylabel('Acceleration')
    plt.legend()

    # Aligned data plot
    plt.subplot(1, 2, 2)
    plt.plot(aligned_data.index, aligned_data['acc_x'], label='acc_x (Aligned)', color='r')
    plt.plot(aligned_data.index, aligned_data['acc_y'], label='acc_y (Aligned)', color='g')
    plt.plot(aligned_data.index, aligned_data['acc_z'], label='acc_z (Aligned)', color='b')
    plt.title(f'{title} - Aligned Data')
    plt.xlabel('Time')
    plt.ylabel('Acceleration')
    plt.legend()

    plt.tight_layout()
    plt.show()

def process_and_rotate_dataset(dataset, exercise_name, visualize=False):
    """
    Main function to process and rotate the dataset based on the detected stationary state and gravity alignment.

    Parameters:
        dataset (pd.DataFrame): Input dataset containing accelerometer and gyroscope data.
        exercise_name (str): Name of the exercise for logging and visualization purposes.
    """
    # Ensure the dataset has the required columns
    required_columns = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
    if not all(col in dataset.columns for col in required_columns):
        raise ValueError("DataFrame must contain 'acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z' columns")

    # Extract the accelerometer data from the dataset
    acc_data = dataset[['acc_x', 'acc_y', 'acc_z']].values
    print(f"Exercise: {exercise_name} - Accelerometer data:\n", dataset[['acc_x', 'acc_y', 'acc_z']])

    # Step 1: Calculate the rotation matrix to align the data with gravity
    R_real_ideal_HD = calc_R(acc_data)

    # Step 2: Apply the rotation to align the accelerometer data with gravity
    aligned_df = rotate_with_gaitmap(dataset, R_real_ideal_HD)
    print(f"Exercise: {exercise_name} - Aligned accelerometer data:\n", aligned_df[['acc_x', 'acc_y', 'acc_z']])

    # Step 3: Apply an additional rotation to align the data with the Mobilise-D reference system
    additional_rotation_matrix = np.array([
        [1, 0, 0],    # X' = X
        [0, -1, 0],   # Y' = -Y
        [0, 0, -1]    # Z' = -Z
    ])

    # Convert the additional rotation matrix to a Rotation object
    additional_rotation = R.from_matrix(additional_rotation_matrix)

    # Apply the additional rotation
    final_rotated_df = rotate_with_gaitmap(aligned_df, additional_rotation.as_matrix())
    print(f"Exercise: {exercise_name} - Final rotated accelerometer data:\n", final_rotated_df[['acc_x', 'acc_y', 'acc_z']])

    if visualize:
        plot_data(dataset[['acc_x', 'acc_y', 'acc_z']], aligned_df[['acc_x', 'acc_y', 'acc_z']], f"Exercise: {exercise_name}")

    return final_rotated_df

if __name__ == "__main__":
    subject_id = "003"
    data_path = f'C:/Users/ac4gt/Desktop/Mob-DPipeline/smartphone/test_data/lab/HA/{subject_id}/'

    # Load the dataset using GenericMobilisedDataset
    mobDataset = GenericMobilisedDataset(
        [os.path.join(data_path, "data.mat")],
        test_level_names=["time_measure", "test", "trial"],
        reference_system='INDIP',
        measurement_condition='laboratory',
        reference_para_level='wb',
        parent_folders_as_metadata=None
    )

    # Process the data as usual
    for i, trial in enumerate(mobDataset, start=1):
        short_trial = trial 
        imu_data = short_trial.data_ss 

        # Process the data as usual
        reoriented_data = process_and_rotate_dataset(imu_data, f"Trial {i} Reorientation", visualize=True) # visualize=True if you want to run this script

# %%
