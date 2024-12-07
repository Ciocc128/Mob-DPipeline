#%%
import os
import pandas as pd
import matplotlib.pyplot as plt
from mobgap.data import GenericMobilisedDataset
from mobgap.initial_contacts import IcdShinImproved
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

base_dir = 'C:/Users/ac4gt/Desktop/Mob-DPipeline/smartphone/test_data/lab/HA/'
#base_dir = 'C:/PoliTO/Tesi/mobgap/smartphone/test_data/lab/HA/'
participant_folder = '001'
plot = False

participants = [participant_folder] if participant_folder else os.listdir(base_dir)

for participant_folder in participants:
    participant_path = os.path.join(base_dir, participant_folder)

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

        for trial in mobDataset[3:]:  # Process trials from the 4th one onward skipping standing and data personalization
            detected_ics, imu_data = calculate_icd_shin_output(trial, plot=False)
            reference_ics = trial.reference_parameters_.ic_list

            # Save detected and reference ICs
            all_detected_ics[trial.group_label] = detected_ics
            all_ref_ics[trial.group_label] = reference_ics

            # Plot ICs
            """imu_data.reset_index(drop=True).plot(y="acc_is")
            plt.plot(reference_ics["ic"], imu_data["acc_is"].iloc[reference_ics["ic"]], "o", label="ref")
            plt.plot(detected_ics["ic"], imu_data["acc_is"].iloc[detected_ics["ic"]], "x", label="icd_ionescu_py")
            plt.legend()
            plt.show()"""

        # Concatenate results
        index_names = ["cohort", "participant_id", "time_measure", "test", "trial"]
        detected_df = pd.concat(all_detected_ics, names=index_names)
        reference_df = pd.concat(all_ref_ics, names=index_names)

        """print(f"Results for participant {participant_folder}")
        print("Detected ICs:")
        display(detected_df)
        print("\n")
        print("Reference ICs:")
        display(reference_df)"""

        combined_ics = {"detected": detected_ics, "reference": reference_ics}
        combined_ics = pd.concat(combined_ics, axis=1).reorder_levels((1, 0), axis=1).drop(columns="lr_label")
        display(combined_ics)

        # calulation of sensitivity and PPV
        # Matching ICs and performance metrics
        tolerance_s = 0.5
        sampling_rate_hz = 100  # Adjust according to your data
        tolerance_samples = as_samples(tolerance_s, sampling_rate_hz)
        per_wb_grouper = create_multi_groupby(detected_ics, reference_ics, groupby="wb_id")
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
        display(metrics_per_wb)

        # Assume matches_per_wb, detected_df, and reference_df are already defined as in your example

        # Filter for rows with match_type "tp" (true positive)
        tp_matches = matches_per_wb[matches_per_wb['match_type'] == 'tp']

        # Initialize an empty list to store rows for the new combined DataFrame
        combined_data = []

        # Iterate over each true positive match
        for index, row in tp_matches.iterrows():
            wb_id = index[0]  # Extract wb_id from the index
            detected_idx = row['ic_id_detected'][1]  # Get the index in detected_df
            reference_idx = row['ic_id_reference'][1]  # Get the index in reference_df

            # Extract the rows from detected_df and reference_df
            detected_ic = detected_df.loc[(wb_id, detected_idx), 'ic']['detected']
            reference_ic = reference_df.loc[(wb_id, reference_idx), 'ic']['reference']

            # Append the data to the combined_data list
            combined_data.append({
                'wb_id': wb_id,
                'step_id_detected': detected_idx,
                'ic_detected': detected_ic,
                'step_id_reference': reference_idx,
                'ic_reference': reference_ic
            })

        # Convert the combined_data list into a DataFrame
        combined_df = pd.DataFrame(combined_data)

        # Display the combined DataFrame
        display(combined_df)

# %%
import os
import pandas as pd
import matplotlib.pyplot as plt
from mobgap.data import GenericMobilisedDataset
from mobgap.initial_contacts import IcdShinImproved
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

base_dir = 'C:/Users/ac4gt/Desktop/Mob-DPipeline/smartphone/test_data/lab/HA/'
participant_folder = '001'
plot = False
index_names = ["cohort", "participant_id", "time_measure", "test", "trial"]

participants = [participant_folder] if participant_folder else os.listdir(base_dir)

for participant_folder in participants:
    participant_path = os.path.join(base_dir, participant_folder)

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
        all_metrics_per_wb = {}
        detected_tp = {}
        reference_tp = {}

        for trial in mobDataset[3:]:  # Process trials from the 4th one onward skipping standing and data personalization
            detected_ics, imu_data = calculate_icd_shin_output(trial, plot=False)
            reference_ics = trial.reference_parameters_.ic_list

            # Save detected and reference ICs
            all_detected_ics[trial.group_label] = detected_ics
            all_ref_ics[trial.group_label] = reference_ics

            # Calculate matches_per_wb and metrics_per_wb for each trial
            tolerance_s = 0.2
            sampling_rate_hz = 100  # Adjust according to your data
            tolerance_samples = as_samples(tolerance_s, sampling_rate_hz)
            per_wb_grouper = create_multi_groupby(detected_ics, reference_ics, groupby="wb_id")

            # Calculate matches per wb
            matches_per_wb = per_wb_grouper.apply(
                lambda df1, df2: categorize_ic_list(
                    ic_list_detected=df1,
                    ic_list_reference=df2,
                    tolerance_samples=tolerance_samples,
                    multiindex_warning=False,
                )
            )
            display(matches_per_wb)

            # Calculate metrics per wb
            metrics_per_wb = matches_per_wb.groupby(level="wb_id").apply(
                lambda df_: pd.Series(calculate_matched_icd_performance_metrics(df_))
            )
            display(metrics_per_wb)

            # Add trial and participant information to metrics for traceability
            all_metrics_per_wb[trial.group_label] = metrics_per_wb

            tp_matches = matches_per_wb[matches_per_wb['match_type'] == 'tp']

            # Convert `trial.group_label` to a tuple for indexing
            group_label_tuple = tuple([getattr(trial.group_label, attr) for attr in index_names])

            all_detected_idx = []
            all_reference_idx = []
            all_wb_id = []

            for index, row in tp_matches.iterrows():
                wb_id = index[0]
                detected_idx = row['ic_id_detected'][1]
                reference_idx = row['ic_id_reference'][1]

                # Append the indices to their respective lists
                all_wb_id.append(wb_id)
                all_detected_idx.append(detected_idx)
                all_reference_idx.append(reference_idx)
 
            detected_tp[group_label_tuple] = detected_ics.loc[(all_wb_id, all_detected_idx), 'ic']
            reference_tp[group_label_tuple] = reference_ics.loc[(all_wb_id, all_reference_idx), 'ic']

            # Combine detected and reference ICs
            detected_df_tp = pd.concat(detected_tp, names=index_names)
            reference_df_tp = pd.concat(reference_tp, names=index_names)
            display(detected_df_tp)
            display(reference_df_tp)

            # Reset the index to prepare for row-wise alignment
            detected_df_tp_reset = detected_df_tp.reset_index()
            reference_df_tp_reset = reference_df_tp.reset_index()

            # Shift the step_id in the reference DataFrame by 1 for row-wise alignment
            reference_df_tp_reset['step_id'] -= 1
            display(reference_df_tp_reset)

            # Concatenate the DataFrames along columns, aligning by the row index
            combined_df = pd.concat([detected_df_tp_reset, reference_df_tp_reset['ic']], axis=1, names=index_names)
            combined_df.rename(columns={('ic'): 'ic_detected', ('ic'): 'ic_reference'}, inplace=True)
            # Display the resulting combined DataFrame
            display(combined_df)


            break

            # Plot ICs if necessary
            """imu_data.reset_index(drop=True).plot(y="acc_is")
            plt.plot(reference_ics["ic"], imu_data["acc_is"].iloc[reference_ics["ic"]], "o", label="ref")
            plt.plot(detected_ics["ic"], imu_data["acc_is"].iloc[detected_ics["ic"]], "x", label="icd_ionescu_py")
            plt.legend()
            plt.show()"""

        # Concatenate results for detected and reference ICs across trials if needed
        metrics_df = pd.concat(all_metrics_per_wb, names=index_names)
        detected_df = pd.concat(all_detected_ics, names=index_names)
        reference_df = pd.concat(all_ref_ics, names=index_names)


        # Display final concatenated DataFrames for reference
        #display(detected_df)
        #display(reference_df)
        display(metrics_df)

# %%
