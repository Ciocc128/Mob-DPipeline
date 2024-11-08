#%% Import necessary libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
from mobgap.data import GenericMobilisedDataset
from mobgap.initial_contacts import IcdIonescu
from mobgap.utils.df_operations import apply_transformations, create_multi_groupby
from mobgap.pipeline.evaluation import ErrorTransformFuncs as E
from mobgap.utils.df_operations import apply_aggregations, CustomOperation
from mobgap.pipeline.evaluation import CustomErrorAggregations as A
from IPython.display import display
from mobgap.utils.conversions import to_body_frame, as_samples
from mobgap.pipeline import GsIterator
from mobgap.initial_contacts.evaluation import categorize_ic_list, calculate_matched_icd_performance_metrics

# Helper function to calculate ICD Ionescu output for a single trial
def calculate_icd_ionescu_output(single_test_data):
    imu_data = to_body_frame(single_test_data.data_ss)
    sampling_rate_hz = single_test_data.sampling_rate_hz
    reference_wbs = single_test_data.reference_parameters_.wb_list

    iterator = GsIterator()
    for (gs, data), result in iterator.iterate(imu_data, reference_wbs):
        result.ic_list = IcdIonescu().detect(data, sampling_rate_hz=sampling_rate_hz).ic_list_

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
        detected_ics, imu_data = calculate_icd_ionescu_output(trial)
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

# Function to process all participants in the base directory
def process_all_participants(base_path, plot_ics=False):
    # Iterate over each participant folder
    for participant_folder in os.listdir(base_path):
        participant_path = os.path.join(base_path, participant_folder)

        # Check if folder contains required data.mat file
        if os.path.isdir(participant_path) and "data.mat" in os.listdir(participant_path):
            print(f"Processing participant: {participant_folder}")
            
            # Process participant data
            detected_ics, reference_ics = process_participant(participant_folder, participant_path, plot_ics=plot_ics)
            
            # Error calculation
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
            )

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
            metrics_per_wb = matches_per_wb.groupby(level="wb_id").apply(
                lambda df_: pd.Series(calculate_matched_icd_performance_metrics(df_))
            )

            # Adding calculation of PPV
            metrics_per_wb["PPV"] = matches_per_wb["TP"] / (matches_per_wb["TP"] + matches_per_wb["FP"])

            # Save results to CSV
            results_folder = os.path.join(participant_path, "results")
            if not os.path.exists(results_folder):
                os.makedirs(results_folder)

            combined_ics_with_errors.to_csv(os.path.join(results_folder, f"ic_errors_subject_{participant_folder}.csv"))
            agg_results.to_csv(os.path.join(results_folder, f"ic_agg_results_subject_{participant_folder}.csv"))
            metrics_per_wb.to_csv(os.path.join(results_folder, f"icd_performance_metrics_subject_{participant_folder}.csv"))

            print(f"Results for participant {participant_folder} saved in {results_folder}")

# Define the base directory containing participant folders
base_dir = 'C:/Users/ac4gt/Desktop/Mob-DPipeline/smartphone/test_data/lab/HA/'

# Run the processing for all participants
process_all_participants(base_dir, plot_ics=False)
