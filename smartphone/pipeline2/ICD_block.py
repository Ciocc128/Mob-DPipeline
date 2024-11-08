"""
# This script performs the detection, comparison, and evaluation of initial contact (IC) points in walking patterns
# using the IcdIonescu algorithm. The script processes data from a subject's walking trial, compares the detected ICs 
# with reference ICs, calculates errors (including absolute and relative errors), aggregates results, and computes 
# performance metrics such as precision, recall, and F1-score.

# The main steps are:
# 1. Load the walking trial data for a subject and run the IcdIonescu algorithm to detect IC points.
# 2. Compare the detected ICs with reference ICs provided in the dataset.
# 3. Plot the ICs on accelerometer data to visually verify the detection.
# 4. Calculate error metrics (absolute error, relative error) between detected and reference ICs.
# 5. Apply various aggregations to summarize errors and calculate overall performance.
# 6. Match ICs within walking bouts and compute performance metrics such as precision, recall, and F1-score.
# 7. Save the error analysis and performance results to CSV files for further analysis.

"""
#%%
import os
import pandas as pd
import matplotlib.pyplot as plt
from mobgap.data import GenericMobilisedDataset
from mobgap.initial_contacts import IcdIonescu
from mobgap.utils.df_operations import apply_transformations, create_multi_groupby
from mobgap.pipeline.evaluation import  ErrorTransformFuncs as E
from mobgap.utils.df_operations import apply_aggregations, CustomOperation
from mobgap.pipeline.evaluation import CustomErrorAggregations as A
from IPython.display import display
from mobgap.utils.conversions import to_body_frame
from mobgap.pipeline import GsIterator
from mobgap.utils.conversions import as_samples
from mobgap.initial_contacts.evaluation import categorize_ic_list, calculate_matched_icd_performance_metrics

# Define subject ID and data path
subject_id = "003"
data_path = f'C:/Users/ac4gt/Desktop/Mob-DPipeline/smartphone/test_data/lab/HA/{subject_id}/'

# Load the dataset
mobDataset = GenericMobilisedDataset(
    [data_path + "data.mat"],
    test_level_names=["time_measure", "test", "trial"],
    reference_system='INDIP',
    measurement_condition='laboratory',
    reference_para_level='wb',
    parent_folders_as_metadata=["cohort", "participant_id"]
)

def calculate_icd_ionescu_output(single_test_data):
    """Calculate the ICD Ionescu output for one sensor from the test data."""
    imu_data = to_body_frame(single_test_data.data_ss)
    sampling_rate_hz = single_test_data.sampling_rate_hz
    reference_wbs = single_test_data.reference_parameters_.wb_list

    iterator = GsIterator()
    for (gs, data), result in iterator.iterate(imu_data, reference_wbs):
        result.ic_list = (
            IcdIonescu()
            .detect(data, sampling_rate_hz=sampling_rate_hz)
            .ic_list_
        )

    det_ics = iterator.results_.ic_list
    return det_ics, imu_data

def load_reference(single_test_data):
    """Load the reference initial contacts from the test data."""
    ref_ics = single_test_data.reference_parameters_.ic_list
    return ref_ics

# Initialize results storage for ic parameters and reference data
all_detected_ics = {}
all_ref_ics = {}

for trial in mobDataset[3:]:
    detected_ics, imu_data = calculate_icd_ionescu_output(trial)
    reference_ics = load_reference(trial)

    # Save reference parameters for comparison later
    all_ref_ics[trial.group_label] = reference_ics

    # Save ic for each trial
    all_detected_ics[trial.group_label] = detected_ics

    imu_data.reset_index(drop=True).plot(y = "acc_is")
    plt.plot(reference_ics["ic"], imu_data["acc_is"].iloc[reference_ics["ic"]], "o", label="ref")
    plt.plot(
        detected_ics["ic"],
        imu_data["acc_is"].iloc[detected_ics["ic"]],
        "x",
        label="icd_ionescu_py",
    )
    plt.legend()
    plt.show()

index_names = ["cohort", "participant_id", "time_measure", "test", "trial"]
all_detected_ics = pd.concat(all_detected_ics, names=index_names)
all_ref_ics = pd.concat(all_ref_ics, names=index_names)
display(all_detected_ics)
display(all_ref_ics)

#%% Calculate error metrics for the detected ICs
# Define error configuration and calculate errors (only abs_error)
combined_ics = {"detected": all_detected_ics, "reference": all_ref_ics}
combined_ics = pd.concat(combined_ics, axis=1).reorder_levels((1, 0), axis=1).drop(columns="lr_label")
display(combined_ics)

errors = [
    ("ic", [E.error, E.abs_error, E.rel_error])
]

# Apply error transformations
errors = apply_transformations(combined_ics, errors)

# Concatenate matches with errors
combined_ics_with_errors = pd.concat([combined_ics, errors], axis=1).sort_index(axis=1)

multiindex_column_order = [
    ('ic', 'detected'),
    ('ic', 'reference'),
    ('ic', 'error'),
    ('ic', 'abs_error')
]

combined_ics_with_errors = combined_ics_with_errors.reindex(columns=multiindex_column_order)
display(combined_ics_with_errors)

# %% Aggregate errors
aggregation = [
    *[(("ic", o), ["mean", "std"]) for o in ["error", "abs_error"]],
    CustomOperation(
        identifier="ic",
        function=A.icc,
        column_name=("ic", "all"),
    )
]

agg_results = (
    apply_aggregations(combined_ics_with_errors.dropna(), aggregation)
    .rename_axis(index=["aggregation", "metric", "origin"])
    .reorder_levels(["metric", "origin", "aggregation"])
    .sort_index(level=0)
    .to_frame("values")
)

display(agg_results)
# %% Matching ICs between detected and reference lists
# Define tolerance in seconds for matching (for example, 0.2 seconds)
tolerance_s = 0.2

# Convert tolerance to samples using the sampling rate from the dataset
sampling_rate_hz = 100  # Adjust according to your data
tolerance_samples = as_samples(tolerance_s, sampling_rate_hz)

# Group the detected and reference ICs by 'wb_id' for comparison
per_wb_grouper = create_multi_groupby(all_detected_ics, all_ref_ics, groupby="wb_id")

# Apply the matching function for each walking bout (wb_id)
matches_per_wb = per_wb_grouper.apply(
    lambda df1, df2: categorize_ic_list(
        ic_list_detected=df1,
        ic_list_reference=df2,
        tolerance_samples=tolerance_samples,
        multiindex_warning=False,
    )
)

# Display the matches per walking bout
display(matches_per_wb)

#%% Calculate performance metrics (precision, recall, F1-score) across all walking bouts
metrics_all = calculate_matched_icd_performance_metrics(matches_per_wb)
print("Overall Performance Metrics:")
print(pd.Series(metrics_all))

# Calculate performance metrics per walking bout
metrics_per_wb = matches_per_wb.groupby(level="wb_id").apply(
    lambda df_: pd.Series(calculate_matched_icd_performance_metrics(df_))
)

# Display performance metrics per walking bout
display(metrics_per_wb)

#%% Save the evaluation metrics to a CSV file

# Create a results folder if it doesn't exist
results_folder = os.path.join(data_path, "results")
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

ic_with_errors_path = os.path.join(results_folder, f"ic_errors_subject_{subject_id}.csv")
combined_ics_with_errors.to_csv(ic_with_errors_path)

agg_results_path = os.path.join(results_folder, f"ic_agg_results_subject_{subject_id}.csv")
agg_results.to_csv(agg_results_path)

metrics_csv_path = os.path.join(results_folder, f"icd_performance_metrics_subject_{subject_id}.csv")
metrics_per_wb.to_csv(metrics_csv_path)
print(f"Performance metrics per walking bout saved to {metrics_csv_path}")
