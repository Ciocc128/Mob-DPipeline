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

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def truncate_to_decimals(x):
    if isinstance(x, float):  # Applica solo ai numeri float
        return round(x, 2)
    elif isinstance(x, (tuple, list)):  # Se è un tuple o una lista, tronca ricorsivamente
        return type(x)(truncate_to_decimals(i) for i in x)
    return x  # Mantieni gli altri valori inalterati


# Helper function to save DataFrames
def save_results_to_folder(dataframes, participant_folder, filenames):
    results_dir = os.path.join(participant_folder, "results")
    os.makedirs(results_dir, exist_ok=True)  # Create the results directory if it doesn't exist
    
    for df, filename in zip(dataframes, filenames):
        file_path = os.path.join(results_dir, filename)
        df.to_csv(file_path, index=True)  # Save with index for traceability
        print(f"Saved {filename} to {file_path}")

base_dir = 'C:/Users/ac4gt/Desktop/Mob-DPipeline/smartphone/test_data/lab/HA/'
#participant_folder = False
participant_folder = '011'
index_names = ["cohort", "participant_id", "time_measure", "test", "trial", "wb_id"]
participants = [participant_folder] if participant_folder else os.listdir(base_dir)

all_subj_with_errors = []

# initialize the pipeline
haPipeline = MobilisedPipelineHealthy()

# Main loop for processing participants
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

        all_ref_parameters = {}
        per_wb_paras = {}

        for test in tqdm(mobDataset):
            try:
                all_ref_parameters[test.group_label] = test.reference_parameters_.wb_list
            except ValueError as e:
                print(f"Skipping test {test.group_label} due to missing reference data.")

        all_ref_parameters = pd.concat(all_ref_parameters, names=index_names)
        all_ref_parameters.columns = all_ref_parameters.columns.str.lstrip("avg_")

        # Run pipeline and collect per-walking bout parameters
        per_wb_paras = {}
        for trial in tqdm(mobDataset[:3]):
            trial.set_params
            pipe = haPipeline.clone().safe_run(trial)
            if not pipe.per_wb_parameters_.empty:
                per_wb_paras[trial.group_label] = pipe.per_wb_parameters_

        all_per_wb_parameters = pd.concat(per_wb_paras, names=index_names).drop(columns=["rule_obj","rule_name"])

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
                overlap_threshold=overlap_th, #controlla nel paper quanto overlap c'è (forse 0.8) riporto entrambi sia 0.8 che 0.6
                multiindex_warning=False
            )
        )

        gs_matches = get_matching_intervals(
            metrics_detected=all_per_wb_parameters,
            metrics_reference=all_ref_parameters,
            matches=gs_tp_fp_fn,
        )

        # Calculate errors
        # Define error configuration and calculate errors
        error_config = [
            ("walking_speed_mps", [E.error, E.abs_error, E.rel_error, E.abs_rel_error]),  # Walking speed should use absolute, relative, and absolute relative errors.
        ]

        errors = apply_transformations(gs_matches, error_config)

        # Concatenate the matches with errors
        gs_matches_with_errors = pd.concat([gs_matches, errors], axis=1).sort_index(axis=1)
        print(f'Error metrics for detected walking bouts with overlap th of {overlap_th} :\n')
        display(gs_matches_with_errors)
        all_subj_with_errors.append(gs_matches_with_errors)

        # Aggregation
        agg_results = (
            apply_aggregations(gs_matches_with_errors.dropna(), get_default_error_aggregations(), missing_columns="skip")
            .rename_axis(index=["aggregation", "metric", "origin"])
            .reorder_levels(["metric", "origin", "aggregation"])
            .sort_index(level=0)
            .to_frame("values")
        )

        agg_results_truncated = agg_results.map(truncate_to_decimals)

        # Display aggregated results
        display(agg_results_truncated)

        # Save results for the participant
        save_results_to_folder(
            dataframes=[gs_matches_with_errors, agg_results_truncated],
            participant_folder=participant_path,
            filenames=[f"ws_with_errors_{participant_folder}.csv", f"ws_agg_results_{participant_folder}.csv"]
        )

#%% Run if you want to process all participants
global_results = pd.concat(all_subj_with_errors)

# Calculate agg_results for the global_combined_tp_with_errors
global_aggregation = [
    *[(("walking_speed_mps", o), ["mean", A.quantiles]) for o in ["abs_error", "abs_rel_error"]],
    *[(("walking_speed_mps", o), ["mean", A.loa]) for o in ["error", "rel_error"]],
    *[CustomOperation(identifier="walking_speed_mps", function=A.icc, column_name=("walking_speed_mps", "all"))],
    CustomOperation(identifier=None, function=A.n_datapoints, column_name=("all", "all")),
]
global_agg_results = (
    apply_aggregations(global_results.dropna(), global_aggregation)
    .rename_axis(index=["aggregation", "metric", "origin"])
    .reorder_levels(["metric", "origin", "aggregation"])
    .sort_index(level=0)
    .to_frame("values")
)

global_agg_results_truncated = global_agg_results.map(truncate_to_decimals)

# Save the global results
results_path = os.path.join(base_dir, "CohortResults")
os.makedirs(results_path, exist_ok=True)
global_results.to_csv(os.path.join(results_path, "inLab_ws_with_errors.csv"))
global_agg_results_truncated.to_csv(os.path.join(results_path, "inLab_ws_agg_results.csv"))

# %%
