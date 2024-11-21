"""
Cadence Evaluation Script with Shin Algorithm for Initial Contact Detection

This script detects initial contacts using the Shin algorithm and evaluates cadence detection based on these detected ICs.
It performs the following steps for each participant:
1. Load the dataset.
2. Detect initial contacts using the `IcdShinImproved` algorithm.
3. Calculate cadence from the detected initial contacts.
4. Compare the detected cadence with the reference cadence using error metrics (absolute and relative errors).
5. Aggregate the errors and compute metrics (e.g., ICC).
6. Save the results (cadence errors and aggregated metrics) to CSV files in a results folder.
"""


#%% Import necessary libraries
from mobgap.cadence import CadFromIcDetector
from mobgap.data import GenericMobilisedDataset
from mobgap.pipeline import GsIterator
from mobgap.initial_contacts import IcdShinImproved, refine_gs
from mobgap.utils.conversions import to_body_frame
import pandas as pd
import os
from IPython.display import display
from mobgap.pipeline.evaluation import ErrorTransformFuncs as E
from mobgap.utils.df_operations import apply_transformations, apply_aggregations, CustomOperation
from mobgap.pipeline.evaluation import CustomErrorAggregations as A

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Helper function to calculate ICD Shin Improved output for a single trial
def calculate_icd_shin_output(single_test_data):
    imu_data = to_body_frame(single_test_data.data_ss)
    sampling_rate_hz = single_test_data.sampling_rate_hz
    reference_wbs = single_test_data.reference_parameters_.wb_list

    iterator = GsIterator()
    for (gs, data), result in iterator.iterate(imu_data, reference_wbs):
        result.ic_list = IcdShinImproved().detect(data, sampling_rate_hz=sampling_rate_hz).ic_list_

    det_ics = iterator.results_.ic_list
    return det_ics, imu_data"""

# Function to process a single participant using Shin IC detection
def process_participant(participant_id, data_path):
    print(f"Starting evaluation for participant {participant_id}")

    # Load the dataset for the participant
    mobDataset = GenericMobilisedDataset(
        [os.path.join(data_path, "data.mat")],
        test_level_names=["time_measure", "test", "trial"],
        parent_folders_as_metadata=["cohort", "participant_id"],
        reference_system='INDIP',
        measurement_condition='laboratory',
        reference_para_level='wb',
    )
    iterator = GsIterator()
    cad_from_ic_detector = CadFromIcDetector(IcdShinImproved())

    all_detected_cad = {}
    all_ref_cad = {}

    for trial in mobDataset[3:]:  # Process trials from the 4th one onward
        reference_gs = trial.reference_parameters_relative_to_wb_.wb_list

        # Calculate ICs using Shin Algorithm
        # Calculate cadence from detected IC

        # Calculate cadence from detected ICs
        cad_from_ic = CadFromIc()

        gs_id = reference_gs.index[0]
        data_in_gs = trial.data_ss.iloc[
            reference_gs.start.iloc[0]:reference_gs.end.iloc[0]
        ]
        display(data_in_gs)
        ics_in_gs = detected_ics[["ic"]].loc[gs_id]
        display(ics_in_gs)

        cad_from_ic.calculate(
            data_in_gs,
            initial_contacts=ics_in_gs,
            sampling_rate_hz=trial.sampling_rate_hz,
        )
        display(cad_from_ic.cadence_per_sec_)

        all_detected_cad[trial.group_label] = cad_from_ic.cadence_per_sec_#["cadence_spm"].mean()
        all_ref_cad[trial.group_label] = reference_gs[["avg_cadence_spm"]].rename(columns={"avg_cadence_spm": "cadence_spm"})
        break

    # Concatenate results
    index_names = ["cohort", "participant_id", "time_measure", "test", "trial"]
    detected_df = pd.concat(all_detected_cad, names=index_names)
    reference_df = pd.concat(all_ref_cad, names=index_names)

    # Error calculation
    combined_cad = {"detected": detected_df, "reference": reference_df}
    combined_cad = pd.concat(combined_cad, axis=1).reorder_levels((1, 0), axis=1)
    display(combined_cad)
    
    errors = [("cadence_spm", [E.abs_error, E.rel_error])]
    cad_errors = apply_transformations(combined_cad, errors)
    combined_cad_with_errors = pd.concat([combined_cad, cad_errors], axis=1)
    display(combined_cad_with_errors)

    # Aggregation
    aggregation = [
        *[(("cadence_spm", o), ["mean", "std"]) for o in ["abs_error", "rel_error"]],
        CustomOperation(identifier="cadence_spm", function=A.icc, column_name=("cadence_spm", "all"))
    ]

    agg_results = (
        apply_aggregations(combined_cad_with_errors, aggregation)
        .rename_axis(index=["aggregation", "metric", "origin"])
        .reorder_levels(["metric", "origin", "aggregation"])
        .sort_index(level=0)
        .to_frame("values")
    )
    display(agg_results)

    # Save results to CSV
    results_folder = os.path.join(data_path, "results")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    combined_cad_with_errors.to_csv(os.path.join(results_folder, f"cadence_errors_subject_{participant_id}.csv"))
    agg_results.to_csv(os.path.join(results_folder, f"cadence_agg_results_subject_{participant_id}.csv"))

    print(f"Results for participant {participant_id} saved in {results_folder}")

# Main function to process all participants
def process_all_participants(base_path):
    for participant_folder in os.listdir(base_path):
        participant_path = os.path.join(base_path, participant_folder)
        
        if os.path.isdir(participant_path) and "data.mat" in os.listdir(participant_path):
            process_participant(participant_folder, participant_path)

# Define the base directory containing participant folders
#base_dir = 'C:/Users/ac4gt/Desktop/Mob-DPipeline/smartphone/test_data/lab/HA/'
base_dir = "C:/PoliTO/Tesi/mobgap/smartphone/test_data/lab/HA/"
# Run the processing for all participants
process_all_participants(base_dir)

# %%
