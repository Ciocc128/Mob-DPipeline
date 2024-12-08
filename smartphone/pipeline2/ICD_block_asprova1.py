
#%% Import necessary libraries
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

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Helper function to calculate ICD Ionescu output for a single trial
def calculate_icd_shin_output(single_test_data):
    imu_data = to_body_frame(single_test_data.data_ss)
    sampling_rate_hz = single_test_data.sampling_rate_hz
    reference_wbs = single_test_data.reference_parameters_.wb_list

    iterator = GsIterator()
    for (gs, data), result in iterator.iterate(imu_data, reference_wbs):
        result.ic_list = IcdShinImproved().detect(data, sampling_rate_hz=sampling_rate_hz).ic_list_

    det_ics = iterator.results_.ic_list
    return det_ics, imu_data

# Helper function to load reference initial contacts
def load_reference(single_test_data):
    return single_test_data.reference_parameters_.ic_list

# Function to process each participant
def process_participant(participant_id, data_path, plot_ics=False):
    mobDataset = GenericMobilisedDataset(
        [os.path.join(data_path, "data.mat")],
        test_level_names=["time_measure", "test", "trial"],
        reference_system='INDIP',
        measurement_condition='laboratory',
        reference_para_level='wb',
        parent_folders_as_metadata=["cohort", "participant_id"]
    )

    all_detected_ics = {}
    all_ref_ics = {}

    for trial in mobDataset[3:]:  # Process trials from the 4th one onward skipping standing and data personalization
        detected_ics, imu_data = calculate_icd_shin_output(trial)
        reference_ics = load_reference(trial)

        # Save detected and reference ICs
        all_detected_ics[trial.group_label] = detected_ics
        all_ref_ics[trial.group_label] = reference_ics

        # Plot ICs if enabled
        if plot_ics:
            imu_data.reset_index(drop=True).plot(y="acc_is")
            plt.plot(reference_ics["ic"], imu_data["acc_is"].iloc[reference_ics["ic"]], "o", label="ref")
            plt.plot(detected_ics["ic"], imu_data["acc_is"].iloc[detected_ics["ic"]], "x", label="icd_ionescu_py")
            plt.legend()
            plt.show()

    # Concatenate results
    index_names = ["cohort", "participant_id", "time_measure", "test", "trial"]
    detected_df = pd.concat(all_detected_ics, names=index_names)
    reference_df = pd.concat(all_ref_ics, names=index_names)

    return detected_df, reference_df

# Function to process all participants or a single participant in the base directory
def process_all_participants(base_path, participant_folder=None, plot_ics=False):
    # If a specific participant folder is given, process only that folder
    if participant_folder:
        participant_path = os.path.join(base_path, participant_folder)

        # Check if the folder contains the required data.mat file
        if os.path.isdir(participant_path) and "data.mat" in os.listdir(participant_path):
            print(f"Processing participant: {participant_folder}")
            
            # Process participant data
            detected_ics, reference_ics = process_participant(participant_folder, participant_path, plot_ics=plot_ics)
            
            """# Error calculation
            combined_ics = {"detected": detected_ics, "reference": reference_ics}
            combined_ics = pd.concat(combined_ics, axis=1).reorder_levels((1, 0), axis=1).drop(columns="lr_label")
            
            errors = [("ic", [E.error, E.abs_error, E.rel_error])]
            errors = apply_transformations(combined_ics, errors)

            # Concatenate matches with errors
            combined_ics_with_errors = pd.concat([combined_ics, errors], axis=1).sort_index(axis=1)
            multiindex_column_order = [
                ('ic', 'detected'), ('ic', 'reference'), ('ic', 'error'), ('ic', 'abs_error'), ('ic', 'rel_error')
            ]
            combined_ics_with_errors = combined_ics_with_errors.reindex(columns=multiindex_column_order)

            # Aggregation
            aggregation = [
                *[(("ic", o), ["mean", "std"]) for o in ["error", "abs_error", "rel_error"]],
                CustomOperation(identifier="ic", function=A.icc, column_name=("ic", "all"))
            ]
            agg_results = (
                apply_aggregations(combined_ics_with_errors.dropna(), aggregation)
                .rename_axis(index=["aggregation", "metric", "origin"])
                .reorder_levels(["metric", "origin", "aggregation"])
                .sort_index(level=0)
                .to_frame("values")
            )"""

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

            print(f"Finished processing participant: {participant_folder}")

    else:
        # Process all participants if no specific folder is provided
        for participant_folder in os.listdir(base_path):
            process_all_participants(base_path, participant_folder, plot_ics)

# Define the base directory containing participant folders
base_dir = 'C:/Users/ac4gt/Desktop/Mob-DPipeline/smartphone/test_data/lab/HA/'

# Run the processing for a specific participant or all participants
# Example for single participant:
process_all_participants(base_dir, participant_folder="011", plot_ics=False)

# Example for all participants:
#process_all_participants(base_dir, plot_ics=False)


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

    print(ic_list_internal)
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

#base_dir = 'C:/Users/ac4gt/Desktop/Mob-DPipeline/smartphone/test_data/lab/HA/'
base_dir = 'C:/PoliTO/Tesi/mobgap/smartphone/test_data/lab/HA/'
participant_folder = '011'
plot = False

participants = [participant_folder] if participant_folder else os.listdir(base_dir)

for participant_folder in os.listdir(base_dir):
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



        

# %%
