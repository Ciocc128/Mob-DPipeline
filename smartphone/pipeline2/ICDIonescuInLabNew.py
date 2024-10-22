#%%
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mobgap.data import GenericMobilisedDataset
from mobgap.initial_contacts import IcdIonescu
from mobgap.utils.df_operations import apply_transformations, create_multi_groupby
from mobgap.pipeline.evaluation import categorize_intervals, get_matching_intervals, ErrorTransformFuncs as E
from mobgap.utils.df_operations import apply_aggregations
from mobgap.pipeline.evaluation import get_default_error_aggregations
from IPython.display import display
from mobgap.utils.conversions import to_body_frame
from mobgap.pipeline import GsIterator
from mobgap.utils.conversions import as_samples
from mobgap.initial_contacts.evaluation import categorize_ic_list, calculate_matched_icd_performance_metrics

# Define subject ID and data path
subject_id = "003"
data_path = f'C:/Users/ac4gt/Desktop/Mob-DPipeline/smartphone/test_data/lab/HA/{subject_id}/'

# Create a results folder if it doesn't exist
results_folder = os.path.join(data_path, "results")
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# Load the dataset
mobDataset = GenericMobilisedDataset(
    [data_path + "data.mat"],
    test_level_names=["time_measure", "test", "trial"],
    reference_system='INDIP',
    measurement_condition='laboratory',
    reference_para_level='wb',
    parent_folders_as_metadata=["cohorts", "participant_id"]
)

# Initialize dictionaries to store reference and detected ICs
ref_ics_dict = {}
detected_ics_dict = {}

# Initialize GsIterator
iterator = GsIterator()

# Process the trials from the 4th to the last one
for trial in mobDataset[3:]:
    imu_data = to_body_frame(trial.data_ss)
    reference_wbs = trial.reference_parameters_.wb_list
    ref_ics = trial.reference_parameters_.ic_list
    iterator = GsIterator()

    # Store trial metadata
    trial_params = trial.get_params()
    trial_metadata = trial.group_label  # Use the full GenericMobilisedDatasetGroupLabel
    time_measure = trial_params['subset_index'].iloc[0]['time_measure']
    test = trial_params['subset_index'].iloc[0]['test']
    trial_name = trial_params['subset_index'].iloc[0]['trial']
    
    # Use the iterator to get detected ICs for each gait sequence (GS)
    for (gs, data), result in iterator.iterate(imu_data, reference_wbs):
        result.ic_list = (
            IcdIonescu().detect(data, sampling_rate_hz=trial.sampling_rate_hz).ic_list_
        )
    
    detected_ics = iterator.results_.ic_list
    # Store detected ICs in a dictionary with trial metadata
    detected_ics_dict[trial_metadata] = detected_ics
    ref_ics_dict[trial_metadata] = ref_ics

# Debugging: Check the structure of the dictionaries before concatenation
print("Detected ICs Dictionary Structure:")
for key, df in detected_ics_dict.items():
    print(f"{key}: {df.shape}")

print("Reference ICs Dictionary Structure:")
for key, df in ref_ics_dict.items():
    print(f"{key}: {df.shape}")

# Adjust index names based on the full metadata structure (5 levels)
index_names = ["cohorts", "participant_id", "time_measure", "test", "trial", "wb_id"]

# Convert dictionaries to DataFrames
try:
    all_ref_ics = pd.concat(ref_ics_dict, names=index_names)
    all_detected_ics = pd.concat(detected_ics_dict, names=index_names)
    print("DataFrames concatenated successfully.")
except Exception as e:
    print(f"Error during concatenation: {e}")

# Display the concatenated reference and detected ICs
display(all_ref_ics)
display(all_detected_ics)

# Save the reference and detected ICs to CSV files
ref_ics_csv_path = os.path.join(results_folder, f"ref_ics_subject_{subject_id}.csv")
detected_ics_csv_path = os.path.join(results_folder, f"detected_ics_subject_{subject_id}.csv")
all_ref_ics.to_csv(ref_ics_csv_path)
all_detected_ics.to_csv(detected_ics_csv_path)

print(f"Reference ICs saved to {ref_ics_csv_path}")
print(f"Detected ICs saved to {detected_ics_csv_path}")

# %% Matching ICs between detected and reference lists
# ----------------------------------------------------
groupby_columns = ["participant_id", "time_measure", "test", "trial", "wb_id"]
# Grouping by walking bout ID for matching
per_wb_grouper = create_multi_groupby(
    all_detected_ics, all_ref_ics, groupby=groupby_columns
)

# Define tolerance in seconds (for matching ICs)
tolerance_s = 0.2
tolerance_samples = as_samples(tolerance_s, trial.sampling_rate_hz)

# Apply the matching function for each walking bout
matches_per_wb = per_wb_grouper.apply(
    lambda df1, df2: categorize_ic_list(
        ic_list_detected=df1,
        ic_list_reference=df2,
        tolerance_samples=tolerance_samples,
        multiindex_warning=False,
    )
)

# %% Calculate performance metrics
# -------------------------------
# Calculate performance metrics across all walking bouts
metrics_all = calculate_matched_icd_performance_metrics(matches_per_wb)
print("Performance Metrics Across All Walking Bouts:")
display(pd.Series(metrics_all))

# Calculate performance metrics for each walking bout separately
metrics_per_wb = matches_per_wb.groupby(level="wb_id").apply(
    lambda df_: pd.Series(calculate_matched_icd_performance_metrics(df_))
)

print("Performance Metrics Per Walking Bout:")
display(metrics_per_wb)

# Save the metrics to CSV
metrics_all_csv_path = os.path.join(results_folder, f"icd_metrics_all_subject_{subject_id}.csv")
metrics_per_wb_csv_path = os.path.join(results_folder, f"icd_metrics_per_wb_subject_{subject_id}.csv")

pd.Series(metrics_all).to_csv(metrics_all_csv_path)
metrics_per_wb.to_csv(metrics_per_wb_csv_path)

print(f"ICD metrics across all walking bouts saved to {metrics_all_csv_path}")
print(f"ICD metrics per walking bout saved to {metrics_per_wb_csv_path}")
#%% Perform Evaluation
# Concatenate reference and detected ICs
"""index_names = ["time_measure", "test", "trial", "wb_id"]
all_ref_ics = pd.concat(all_ref_ics, names=index_names)
all_detected_ics = pd.concat(all_detected_ics, names=index_names)

# Compare detected ICs with reference ICs
ic_grouper = create_multi_groupby(
    all_detected_ics,
    all_ref_ics,
    groupby=index_names[:-1],
)

# Categorize intervals
overlap_th = 0.8
ic_tp_fp_fn = ic_grouper.apply(
    lambda det, ref: categorize_intervals(
        gsd_list_detected=det,
        gsd_list_reference=ref,
        overlap_threshold=overlap_th,
        multiindex_warning=False
    )
)

# Get matching intervals for error calculations
ic_matches = get_matching_intervals(
    metrics_detected=all_detected_ics,
    metrics_reference=all_ref_ics,
    matches=ic_tp_fp_fn,
)

# Define error configuration (absolute error of ICs)
error_config = [
    ("ic", [E.abs_error]),   # Absolute error of ICs
]
errors = apply_transformations(ic_matches, error_config)

# Concatenate matches with errors
ic_matches_with_errors = pd.concat([ic_matches, errors], axis=1).sort_index(axis=1)
display(ic_matches_with_errors)

# Save the IC matches with errors as CSV
csv_file_path = os.path.join(results_folder, f"icd_subject_{subject_id}.csv")
ic_matches_with_errors.to_csv(csv_file_path)
print(f"IC matches with errors saved to {csv_file_path}")

#%% Apply aggregation for IC evaluation (focus on abs_error)
aggregation = {
    ("ic", "abs_error"): ["mean", "std"]
}

agg_results = (
    apply_aggregations(ic_matches_with_errors, aggregation, missing_columns="skip")
    .rename_axis(index=["aggregation", "metric", "origin"])
    .reorder_levels(["metric", "origin", "aggregation"])
    .sort_index(level=0)
    .to_frame("values")
)

# Save aggregated results as CSV
agg_csv_file_path = os.path.join(results_folder, f"icd_agg_results_subject_{subject_id}.csv")
agg_results.to_csv(agg_csv_file_path)
print(f"Aggregated results saved to {agg_csv_file_path}")

# Display aggregated results
display(agg_results)

#%% Save JSON Output
data_output_native = convert_to_native_types(data_output)
json_file_path = os.path.join(results_folder, f"icd_output_subject_{subject_id}.json")
with open(json_file_path, "w") as json_file:
    json.dump(data_output_native, json_file, indent=4)
print(f"ICD JSON results saved to {json_file_path}")"""

