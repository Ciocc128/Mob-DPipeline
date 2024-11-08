"""
# This script performs gait sequence detection and evaluation using the GsdAdaptiveIonescu algorithm.
# It processes the walking trial data for a subject, compares detected gait sequences with reference sequences,
# calculates errors, aggregates the results, and evaluates the accuracy of the detection based on several metrics.

# The main steps are:
# 1. Load the walking trial data for a subject and run the GsdAdaptiveIonescu algorithm to detect gait sequences.
# 2. Compare the detected gait sequences with the reference sequences provided in the dataset.
# 3. Plot the gait sequences on accelerometer data for visual verification of the detection.
# 4. Calculate error metrics (e.g., absolute error for start and end of gait sequences).
# 5. Aggregate the errors by calculating mean and standard deviation, and apply advanced metrics like ICC (Intraclass Correlation Coefficient).
# 6. Save the error analysis and aggregated results to CSV files for further evaluation.
"""
#%%
import os
import pandas as pd
import matplotlib.pyplot as plt
from mobgap.data import GenericMobilisedDataset
from mobgap.gait_sequences import GsdAdaptiveIonescu
from mobgap.utils.df_operations import apply_transformations, create_multi_groupby
from mobgap.pipeline.evaluation import categorize_intervals, get_matching_intervals, ErrorTransformFuncs as E
from mobgap.pipeline.evaluation import get_default_error_aggregations
from mobgap.pipeline.evaluation import CustomErrorAggregations as A
from mobgap.utils.df_operations import apply_aggregations, CustomOperation
from IPython.display import display

# Define subject ID and data path
subject_id = "003"
data_path = f'C:/Users/ac4gt/Desktop/Mob-DPipeline/smartphone/test_data/lab/HA/{subject_id}/'

# Helper function to plot results
def plot_gsd_outputs(data, **kwargs):
    fig, ax = plt.subplots()
    ax.plot(data["acc_x"].to_numpy(), label="acc_x")

    color_cycle = iter(plt.rcParams["axes.prop_cycle"])

    y_max = 1.1
    plot_props = [
        {
            "data": v,
            "label": k,
            "alpha": 0.2,
            "ymax": (y_max := y_max - 0.1),
            "color": next(color_cycle)["color"],
        }
        for k, v in kwargs.items()
    ]

    for props in plot_props:
        for gsd in props.pop("data").itertuples(index=False):
            ax.axvspan(gsd.start, gsd.end, label=props.pop("label", None), **props)

    ax.legend()
    return fig, ax


# Load the dataset
mobDataset = GenericMobilisedDataset(
    [data_path + "data.mat"],
    test_level_names=["time_measure", "test", "trial"],
    reference_system='INDIP',
    measurement_condition='laboratory',
    reference_para_level='wb',
    parent_folders_as_metadata=["cohort", "participant_id"]
)

# Process the trials from the 4th to the last one
def run_gsd_analysis(plot_gsd=True):
    for trial in mobDataset[3:]:
        imu_data = trial.data_ss
        reference_data = trial.reference_parameters_.wb_list
        gs_output = GsdAdaptiveIonescu().detect(imu_data, sampling_rate_hz=trial.sampling_rate_hz)
        
        # Save reference parameters for comparison later
        all_ref_parameters[trial.group_label] = reference_data

        # Save per-walking bout parameters for each trial
        per_wb_paras[trial.group_label] = gs_output.gs_list_

        # Plotting (if enabled)
        if plot_gsd:
            fig, ax = plot_gsd_outputs(
                trial.data_ss,
                reference=reference_data,
                detected=gs_output.gs_list_,
            )
            plt.show()  # Show the plot for each trial

# Initialize results storage for per-walking bout parameters and reference data
per_wb_paras = {}
all_ref_parameters = {}

# Call the analysis function
run_gsd_analysis(plot_gsd=True)
#%% Concatenate reference parameters and per-walking bout parameters
index_names = ["cohort", "participant_id", "time_measure", "test", "trial", "wb_id"]
all_ref_parameters = pd.concat(all_ref_parameters, names=index_names)
all_per_wb_parameters = pd.concat(per_wb_paras, names=index_names)

# Compare detected walking bouts with reference
per_trial_participant_grouper = create_multi_groupby(
    all_per_wb_parameters,
    all_ref_parameters,
    groupby=index_names[:-1],
)

# Apply categorization intervals for evaluation
overlap_th = 0.8  # Set overlap threshold
gs_tp_fp_fn = per_trial_participant_grouper.apply(
    lambda det, ref: categorize_intervals(
        gsd_list_detected=det,
        gsd_list_reference=ref,
        overlap_threshold=overlap_th,
        multiindex_warning=False
    )
)

# Get matching intervals for error calculations
gs_matches = get_matching_intervals(
    metrics_detected=all_per_wb_parameters,
    metrics_reference=all_ref_parameters,
    matches=gs_tp_fp_fn,
)

# Define error configuration and calculate errors (only abs_error)
error_config = [
    ("start", [E.abs_error]),   # Start sample absolute error
    ("end", [E.abs_error]),     # End sample absolute error
]
# Apply error transformations
errors = apply_transformations(gs_matches, error_config)

# Concatenate matches with errors
gs_matches_with_errors = pd.concat([gs_matches, errors], axis=1).sort_index(axis=1).drop(columns="wb_id")
multiindex_column_order = [
    # 'start' columns
    ('start', 'detected'),
    ('start', 'reference'),
    ('start', 'abs_error'),
    
    # 'end' columns
    ('end', 'detected'),
    ('end', 'reference'),
    ('end', 'abs_error'),
]
# Reindex the DataFrame to follow the new MultiIndex column order
gs_matches_with_errors = gs_matches_with_errors.reindex(columns=multiindex_column_order)
display(gs_matches_with_errors)
#%% Apply aggregation focusing on 'start' and 'end' parameters
aggregation = [
    (("start", "abs_error"), ["mean", "std"]),
    (("end", "abs_error"), ["mean", "std"]),
    *(
        CustomOperation(identifier=m, function=A.icc, column_name=(m, "all"))
        for m in ["start", "end"]
    )
]

agg_results = (
    apply_aggregations(gs_matches_with_errors, aggregation, missing_columns="skip")
    .rename_axis(index=["aggregation", "metric", "origin"])
    .reorder_levels(["metric", "origin", "aggregation"])
    .sort_index(level=0)
    .to_frame("values")
)

# Display aggregated results
display(agg_results)

# %% Save results

# Create a results folder if it doesn't exist
results_folder = os.path.join(data_path, "results")
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# Save the results as CSV
csv_file_path = os.path.join(results_folder, f"gsd_errors_subject_{subject_id}.csv")
gs_matches_with_errors.to_csv(csv_file_path)
print(f"GS matches with errors saved to {csv_file_path}")

# Save the aggregated results as CSV
agg_csv_file_path = os.path.join(results_folder, f"gsd_agg_results_subject_{subject_id}.csv")
agg_results.to_csv(agg_csv_file_path)
print(f"Aggregated results saved to {agg_csv_file_path}")

# TODO valutare anche le metriche con TP FP FN ecc..