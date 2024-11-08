"""# load the reference parameters of the dataset
all_ref_parameters = {}

for test in tqdm(dataset):
    all_ref_parameters[test.group_label] = test.reference_parameters_.wb_list

all_ref_parameters = pd.concat(all_ref_parameters, names=index_name)
all_ref_parameters.columns = all_ref_parameters.columns.str.lstrip("avg_")
all_ref_parameters

# now I want to compare only the walking bouts that are dected by both the pipeline and the reference system, identifying the ones that have a certain overlap
from mobgap.utils.df_operations import create_multi_groupby
from mobgap.pipeline.evaluation import categorize_intervals, get_matching_intervals

per_trial_participant_day_grouper - create_multi_groupby(
    all_per_wb_parameters,
    all_ref_parameters,
    gtopuby=index_names[:-1],
)

gs_tp_fp_fn = per_trial_participant_day_grouper.apply(
    lambda det, ref: categorize_intervals(
        gsd_list_detected=det,
        gsd_list_reference=ref,
        overlap_threshold=0.6,
        multiindex_warning=False,
    )
)

gs_matches = get_matching_intervals(
    metrics_detected= all_per_wb_parameters,
    metrics_reference=all_ref_parameters,
    matches=gs_tp_fp_fn,
)

gs_matches

# so now I want to calculate the error metrics for the detected walking bouts
from mobgap.pipeline.evaluation import ErrorTransformFuncs as E
from mobgap.utils.df_operations import apply_transformations

error_config = {
    ("cadence_spm", [E.abs_error, E.rel_error]),
    ("stride_length_m", [E.abs_error, E.rel_error, E.abs_rel_error]),
    ("duration_s", [E.abs_error]),
}

errors = apply_transformations(gs_matches, error_config)
errors

# now what I want is to aggregate the error metrics for all the walking bouts
# I concatenate the detected walking bouts with the error metrics
gs_matches_with_errors = pd.concat([gs_matches, errors], axis=1).sort_index(axis=1)
gs_matches_with_errors

from mobgap.pipeline.evaluation import CustomErrorAggregations as A
from mobgap.utils.df_operations import apply_aggregations, CustomOperation

aggregation = {
    *((("stride_length_m", o), ["mean", "std",]) for o in ["error", "abs_error", "rel_error"]),
    *((("cadence_spm", o), ["mean", "std",]) for o in ["abs_error", "rel_error"]),
    (("duration_s", "abs_error"), ["mean", "std"]),
    *(
        CustomOperation(identifier=m, function=A.icc, column_name=(m, "all"))
        for m in ["stride_length_m", "cadence_spm"]
    )
}

agg_results =  (
    apply_aggregations(gs_matches_with_errors, aggregation)
    .rename_axis(index=["aggregation", "metric", "origin"])
    .reorder_levels(["metric", "origin", "aggregation"])
    .sort_index(level=0)
    .to_frame("values")
)

agg_results

# I try to optimize stuff. the code that I have written only considered detected walking bouts but I want also to consider in the evaluation how many walking bouts were missed by the pipeline
"""
#%%
from tqdm.auto import tqdm
from mobgap.data import LabExampleDataset, GenericMobilisedDataset
from mobgap.pipeline import MobilisedPipelineHealthy
from mobgap.utils.df_operations import apply_transformations, create_multi_groupby
from mobgap.pipeline.evaluation import categorize_intervals, get_matching_intervals, get_default_error_transformations, get_default_error_aggregations, ErrorTransformFuncs as E
from mobgap.pipeline.evaluation import CustomErrorAggregations as A
from mobgap.utils.df_operations import apply_aggregations, CustomOperation
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import seaborn as sns
import os

# Define subject ID and data path
subject_id = "003"
data_path = f'C:/Users/ac4gt/Desktop/Mob-DPipeline/smartphone/test_data/lab/HA/{subject_id}/'

print('Starting evaluation for subject', subject_id)

# Load the dataset
mobDataset = GenericMobilisedDataset(
    [data_path + "data.mat"],
    test_level_names=["time_measure", "test", "trial"],
    parent_folders_as_metadata=["cohort", "participant_id"],
    reference_system='INDIP',
    measurement_condition='laboratory',
    reference_para_level='wb',
)


haPipeline = MobilisedPipelineHealthy()

# Load the reference parameters for each trial
# Load the reference parameters, skip if missing
index_names = ["cohort", "participant_id", "time_measure", "test", "trial", "wb_id"]
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

all_ref_parameters = {}

for test in tqdm(mobDataset):
    try:
        all_ref_parameters[test.group_label] = test.reference_parameters_.wb_list
    except ValueError as e:
        print(f"Skipping test {test.group_label} due to missing reference data.")

all_ref_parameters = pd.concat(all_ref_parameters, names=index_names)
all_ref_parameters.columns = all_ref_parameters.columns.str.lstrip("avg_")

print('\nReference parameters:\n')
display(all_ref_parameters)

# Run pipeline and collect per-walking bout parameters
per_wb_paras = {}
for trial in tqdm(mobDataset):
    trial.set_params
    pipe = haPipeline.clone().safe_run(trial)
    if not pipe.per_wb_parameters_.empty:
        per_wb_paras[trial.group_label] = pipe.per_wb_parameters_

all_per_wb_parameters = pd.concat(per_wb_paras, names=index_names).drop(columns=["rule_obj","rule_name"])

print('Per-walking bout parameters:\n')
display(all_per_wb_parameters)

# Compare detected walking bouts with reference
per_trial_participant_grouper = create_multi_groupby(
    all_per_wb_parameters,
    all_ref_parameters,
    groupby=index_names[:-1],
)

overlap_th = 0.8
gs_tp_fp_fn = per_trial_participant_grouper.apply(
    lambda det, ref: categorize_intervals(
        gsd_list_detected=det,
        gsd_list_reference=ref,
        overlap_threshold=0.6, #controlla nel paper quanto overlap c'è (forse 0.8) riporto entrambi sia 0.8 che 0.6
        multiindex_warning=False
    )
)

# fare anche una combined 

# Get matching intervals for further error calculations
gs_matches = get_matching_intervals(
    metrics_detected=all_per_wb_parameters,
    metrics_reference=all_ref_parameters,
    matches=gs_tp_fp_fn,
)

# Define error configuration and calculate errors
error_config = [
    ("start", [E.abs_error]),   # Start time can have absolute error in terms of seconds.
    ("end", [E.abs_error]),     # End time can have absolute error in terms of seconds.
    ("cadence_spm", [E.abs_error, E.rel_error]),  # Cadence (steps per minute) should use absolute and relative errors.
    ("stride_length_m", [E.abs_error, E.rel_error, E.abs_rel_error]),  # Stride length can use absolute, relative, and absolute relative errors.
    ("duration_s", [E.abs_error]),   # Duration can have absolute error in seconds.
    ("walking_speed_mps", [E.abs_error, E.rel_error, E.abs_rel_error]),  # Walking speed should use absolute, relative, and absolute relative errors.
]

# I can also pass the default Mobilise-D error configuration using get_default_error_transformations()
errors = apply_transformations(gs_matches, error_config)

print(f'Error metrics for detected walking bouts with overlap th of {overlap_th} :\n')
display(errors)

# Concatenate the matches with errors
gs_matches_with_errors = pd.concat([gs_matches, errors], axis=1).sort_index(axis=1)
# Define the desired column order following the structure from the error_config
multiindex_column_order = [

    # 'wb_id' columns (only detected and reference)
    ('wb_id', 'detected'),
    ('wb_id', 'reference'),

    # 'start' columns
    ('start', 'detected'),
    ('start', 'reference'),
    ('start', 'abs_error'),
    
    # 'end' columns
    ('end', 'detected'),
    ('end', 'reference'),
    ('end', 'abs_error'),
    
    # 'cadence_spm' columns
    ('cadence_spm', 'detected'),
    ('cadence_spm', 'reference'),
    ('cadence_spm', 'abs_error'),
    ('cadence_spm', 'rel_error'),
    
    # 'stride_length_m' columns
    ('stride_length_m', 'detected'),
    ('stride_length_m', 'reference'),
    ('stride_length_m', 'abs_error'),
    ('stride_length_m', 'rel_error'),
    ('stride_length_m', 'abs_rel_error'),
    
    # 'duration_s' columns
    ('duration_s', 'detected'),
    ('duration_s', 'reference'),
    ('duration_s', 'abs_error'),
    
    # 'walking_speed_m_s' columns
    ('walking_speed_mps', 'detected'),
    ('walking_speed_mps', 'reference'),
    ('walking_speed_mps', 'abs_error'),
    ('walking_speed_mps', 'rel_error'),
    ('walking_speed_mps', 'abs_rel_error'),
    
    # 'n_strides' columns (only detected and reference)
    ('n_strides', 'detected'),
    ('n_strides', 'reference')
]

# Reindex the DataFrame to follow the new MultiIndex column order
gs_matches_with_errors = gs_matches_with_errors.reindex(columns=multiindex_column_order)

# Display the updated DataFrame with the new column order
print(f'Metrics and errors for detected walking bouts:\n')
display(gs_matches_with_errors)

#%% Define aggregations and calculate them
"""aggregation = [
    *((("stride_length_m", o), ["mean", "std"]) for o in ["error", "abs_error", "rel_error"]),
    *((("cadence_spm", o), ["mean", "std"]) for o in ["abs_error", "rel_error"]),
    (("duration_s", "abs_error"), ["mean", "std"]),
    *(
        CustomOperation(identifier=m, function=A.icc, column_name=(m, "all"))
        for m in ["stride_length_m", "cadence_spm"]
    )
]"""
# I can also pass the default Mobilise-D aggregation configuration

agg_results = (
    apply_aggregations(gs_matches_with_errors.dropna(), get_default_error_aggregations(), missing_columns="skip")
    .rename_axis(index=["aggregation", "metric", "origin"])
    .reorder_levels(["metric", "origin", "aggregation"])
    .sort_index(level=0)
    .to_frame("values")
)

# Display aggregated results
display(agg_results)


# %%
# Prepare data for bar plots: Mean Absolute Errors
mean_abs_errors = {
    "cadence_spm": agg_results.loc[("cadence_spm", "abs_error", "mean")].values[0],
    "stride_length_m": agg_results.loc[("stride_length_m", "abs_error", "mean")].values[0],
    "duration_s": agg_results.loc[("duration_s", "abs_error", "mean")].values[0],
    "walking_speed_mps": agg_results.loc[("walking_speed_mps", "abs_error", "mean")].values[0]
}

# Plot Mean Absolute Errors
plt.figure(figsize=(8, 5))
plt.bar(mean_abs_errors.keys(), mean_abs_errors.values(), color='skyblue')
plt.title(f'Mean Absolute Errors for Gait Parameters\n(Subject ID: {subject_id})', fontsize=16)
plt.xlabel('Gait Parameters', fontsize=12)
plt.ylabel('Mean Absolute Error (units depend on parameter)', fontsize=12)
plt.xticks(rotation=45, ha='right')  # Rotate the x-axis labels for better readability
plt.grid(True, axis='y', linestyle='--', alpha=0.7)  # Add a horizontal grid for clarity
plt.tight_layout()
plt.show()

# Prepare data for Intraclass Correlation Coefficients (ICC)
icc_values = {
    "cadence_spm": agg_results.loc[("cadence_spm", "all", "icc")].values[0][0],  # ICC mean value
    "stride_length_m": agg_results.loc[("stride_length_m", "all", "icc")].values[0][0],
    "n_strides": agg_results.loc[("n_strides", "all", "icc")].values[0][0],
    "walking_speed_mps": agg_results.loc[("walking_speed_mps", "all", "icc")].values[0][0]
}

# Plot ICC Values
plt.figure(figsize=(8, 5))
plt.bar(icc_values.keys(), icc_values.values(), color='lightgreen')
plt.title(f'Intraclass Correlation Coefficients (ICC) for Gait Parameters\n(Subject ID: {subject_id})', fontsize=16)
plt.xlabel('Gait Parameters', fontsize=12)
plt.ylabel('ICC Value (0-1 scale)', fontsize=12)
plt.xticks(rotation=45, ha='right')  # Rotate the x-axis labels for better readability
plt.ylim(0, 1)  # ICC ranges between 0 and 1
plt.grid(True, axis='y', linestyle='--', alpha=0.7)  # Add a horizontal grid for clarity
plt.tight_layout()
plt.show()

# Box Plot for Quantiles (Distribution of Errors)
data = [
    agg_results.loc[("cadence_spm", "abs_error", "quantiles")].values[0],
    agg_results.loc[("stride_length_m", "abs_error", "quantiles")].values[0],
    agg_results.loc[("duration_s", "abs_error", "quantiles")].values[0],
    agg_results.loc[("walking_speed_mps", "abs_error", "quantiles")].values[0]
]

fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(data=data)
ax.set_xticklabels(['cadence_spm', 'stride_length_m', 'duration_s', 'walking_speed_mps'])
ax.set_title(f'Distribution of Absolute Errors for Gait Parameters\n(Subject ID: {subject_id})', fontsize=16)
ax.set_xlabel('Gait Parameters', fontsize=12)
ax.set_ylabel('Absolute Error (units depend on parameter)', fontsize=12)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)  # Add a horizontal grid for clarity
plt.tight_layout()
plt.show()


# %%
# Ensure the 'results' subfolder exists
results_folder = os.path.join(data_path, "results")
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# Save gs_matches_with_errors to CSV
gs_matches_with_errors_filepath = os.path.join(results_folder, f"full_pipeline_subject_{subject_id}.csv")
gs_matches_with_errors.to_csv(gs_matches_with_errors_filepath)

# Save agg_results to CSV
agg_results_filepath = os.path.join(results_folder, f"full_pipeline_agg_results_subject_{subject_id}.csv")
agg_results.to_csv(agg_results_filepath)

print(f'Results saved to {results_folder}')
