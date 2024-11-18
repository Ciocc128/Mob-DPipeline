# %%
# %%
import os
import pandas as pd
import matplotlib.pyplot as plt
from mobgap.data import GenericMobilisedDataset
from mobgap.initial_contacts import IcdIonescu, IcdShinImproved
from mobgap.utils.df_operations import apply_transformations, create_multi_groupby
from mobgap.pipeline.evaluation import ErrorTransformFuncs as E
from mobgap.utils.df_operations import apply_aggregations, CustomOperation
from mobgap.pipeline.evaluation import CustomErrorAggregations as A
from IPython.display import display
from mobgap.utils.conversions import to_body_frame, as_samples
from mobgap.pipeline import GsIterator
from mobgap.initial_contacts.evaluation import categorize_ic_list, calculate_matched_icd_performance_metrics
import numpy as np

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Helper function to calculate ICD Shin output for a single trial
def calculate_icd_shin_output(single_test_data, plot):
    imu_data = to_body_frame(single_test_data.data_ss)
    sampling_rate_hz = single_test_data.sampling_rate_hz
    reference_wbs = single_test_data.reference_parameters_.wb_list

    # Initialize iterator for processing
    iterator = GsIterator()
    final_filtered_signal = None  # To store the filtered signal for plotting
    ic_list_internal = []  # To store detected ICs for plotting

    for (gs, data), result in iterator.iterate(imu_data, reference_wbs):
        icd_shin = IcdShinImproved()
        detected = icd_shin.detect(data, sampling_rate_hz=sampling_rate_hz)
        result.ic_list = detected.ic_list_
        final_filtered_signal = icd_shin.final_filtered_signal_  # Store filtered signal
        ic_list_internal = icd_shin.ic_list_internal_  # Internal ICs detected for plotting

    # Convert ic_list_internal to integer indices
    ic_list_internal = np.array(ic_list_internal).astype(int)  # Convert to integer array

    # Detected ICs from iterator
    det_ics = iterator.results_.ic_list

    if plot:
        # Plot filtered signal and detected ICs
        plt.figure(figsize=(12, 6))
        plt.plot(final_filtered_signal, label='Filtered Signal')  # Plot filtered signal
        plt.scatter(ic_list_internal, final_filtered_signal[ic_list_internal], color='red', label='Detected ICs')  # Mark detected ICs
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude')
        plt.title('Filtered Signal with Detected Initial Contacts (ICs)')
        plt.legend()
        plt.show()

    return det_ics, imu_data

# Main function to process participants
base_dir = 'C:/Users/ac4gt/Desktop/Mob-DPipeline/smartphone/test_data/lab/HA/'
participant_folder='011'
plot=False

# If a specific participant is provided, only process that one
participants = [participant_folder] if participant_folder else os.listdir(base_dir)

for participant_folder in participants:
    participant_path = os.path.join(base_dir, participant_folder)

    # Check if the folder contains the required data.mat file
    if os.path.isdir(participant_path) and "data.mat" in os.listdir(participant_path):
        print(f"Processing participant: {participant_folder}")

        # Load the data
        mobDataset = GenericMobilisedDataset(
            [os.path.join(participant_path, "data.mat")],
            test_level_names=["time_measure", "test", "trial"],
            reference_system='INDIP',
            measurement_condition='laboratory',
            reference_para_level='wb',
            parent_folders_as_metadata=["cohort", "participant_id"]
        )

        all_detected_ics = {}
        all_ref_ics = {}

        for trial in mobDataset[3:]:  # Process trials from the 4th one onward
            detected_ics, imu_data = calculate_icd_shin_output(trial, plot=plot)
            reference_ics = trial.reference_parameters_.ic_list

            # Save detected and reference ICs
            all_detected_ics[trial.group_label] = detected_ics
            all_ref_ics[trial.group_label] = reference_ics

        # Concatenate results
        index_names = ["cohort", "participant_id", "time_measure", "test", "trial"]
        detected_df = pd.concat(all_detected_ics, names=index_names)
        reference_df = pd.concat(all_ref_ics, names=index_names)

        display(detected_df)
        display(reference_df)

        # Combine detected and reference ICs
        combined_ics = {"detected": detected_df, "reference": reference_df}
        combined_ics = pd.concat(combined_ics, axis=1).reorder_levels((1, 0), axis=1).drop(columns="lr_label")

        # Calculate sensitivity and PPV
        tolerance_s = 0.5
        sampling_rate_hz = 100
        tolerance_samples = as_samples(tolerance_s, sampling_rate_hz)
        per_wb_grouper = create_multi_groupby(detected_df, reference_df, groupby="wb_id")

        matches_per_wb = per_wb_grouper.apply(
            lambda df1, df2: categorize_ic_list(
                ic_list_detected=df1,
                ic_list_reference=df2,
                tolerance_samples=tolerance_samples,
                multiindex_warning=False,
            )
        )
        display(matches_per_wb)

        metrics_per_wb = matches_per_wb.groupby(level="wb_id").apply(
            lambda df_: pd.Series(calculate_matched_icd_performance_metrics(df_))
        )

        # Calculate the absolute error and add it to the DataFrame
        combined_ics.loc[:, ('ic', 'absolute_error')] = np.abs(combined_ics[('ic', 'detected')] - combined_ics[('ic', 'reference')])

        # Display the DataFrame with absolute error
        display(combined_ics[[('ic', 'detected'), ('ic', 'reference'), ('ic', 'absolute_error')]])

        # Mean Absolute Error (MAE), excluding NaN values
        mae = combined_ics[('ic', 'absolute_error')].mean(skipna=True)
        print("Mean Absolute Error (MAE):", mae)

        # Calculate the percentage of detections within a tolerance range
        within_tolerance = (combined_ics[('ic', 'absolute_error')] <= tolerance_samples).mean(skipna=True) * 100
        print(f"Percentage within ±{tolerance_samples} samples:", within_tolerance, "%")
        print("Metrics per WB:")
        display(metrics_per_wb)

        print("\n\n")
        print("------------------------------------------------------------------------------------------------------------------")
        print("\n\n")


# Define the base directory containing participant folders

#%% Run the processing for a specific participant or all participants
# Example for single participant:
process_participants(base_dir, participant_folder="011", plot=False)

# Example for all participants:
#process_participants(base_dir, plot=False)

# %%
