#%%
from tqdm.auto import tqdm
from mobgap.data import GenericMobilisedDataset
from mobgap.pipeline import MobilisedPipelineHealthy
from mobgap.utils.df_operations import apply_transformations, create_multi_groupby
from mobgap.pipeline.evaluation import categorize_intervals, get_matching_intervals, get_default_error_transformations, get_default_error_aggregations, ErrorTransformFuncs as E
from mobgap.pipeline.evaluation import CustomErrorAggregations as A
from mobgap.utils.df_operations import apply_aggregations, CustomOperation
import pandas as pd
from IPython.display import display
import os
import re

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

def classify_samples(n_samples, detected_intervals, reference_intervals):
    # Initialize all samples as 'tn' (true negative)
    classifications = pd.Series(['tn'] * n_samples)

    # Process detected intervals for tp and fp classifications
    for _, row in detected_intervals.iterrows():
        start = int(row['start'])  # Convert to integer
        end = int(row['end'])      # Convert to integer
        for i in range(start, end + 1):
            # Only classify within bounds
            if i < n_samples:
                if classifications[i] == 'tn':
                    classifications[i] = 'fp'
                elif classifications[i] == 'fn':
                    classifications[i] = 'tp'
            else:
                # Log or print a message for out-of-bounds sample
                print(f"Warning: Detected interval end {row['end']} exceeds sample length {n_samples}")

    # Process reference intervals for fn classifications
    for _, row in reference_intervals.iterrows():
        start = int(row['start'])  # Convert to integer
        end = int(row['end'])      # Convert to integer
        for i in range(start, end + 1):
            # Only classify within bounds
            if i < n_samples:
                if classifications[i] == 'tn':
                    classifications[i] = 'fn'
                elif classifications[i] == 'fp':
                    classifications[i] = 'tp'
            else:
                # Log or print a message for out-of-bounds sample
                print(f"Warning: Reference interval end {row['end']} exceeds sample length {n_samples}")

    return classifications

#base_dir = 'C:/PoliTO/Tesi/mobgap/smartphone/test_data/lab/HA/'
base_dir = 'C:/Users/ac4gt/Desktop/Mob-DPipeline/smartphone/test_data/lab/HA/'
#participant_folder = False
participant_folder = '011'
index_names = ["cohort", "participant_id", "time_measure", "test", "trial", "wb_id"]
participants = [participant_folder] if participant_folder else os.listdir(base_dir)

all_subj_with_errors = []
aggregated_sample_wise_metrics = []

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
        matched_ref = {}
        matched_det = {}
        sample_wise_metrics_per_trial = []

        for test in tqdm(mobDataset):
            try:
                all_ref_parameters[test.group_label] = test.reference_parameters_.wb_list
            except ValueError as e:
                print(f"Skipping test {test.group_label} due to missing reference data.")

        all_ref_parameters = pd.concat(all_ref_parameters, names=index_names)
        all_ref_parameters.columns = all_ref_parameters.columns.str.lstrip("avg_")

        # Run pipeline and collect per-walking bout parameters
        per_wb_paras = {}
        for trial in tqdm(mobDataset[3:]):
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
        display(gs_matches)

        for trial in mobDataset:
            # Extract the group_label from the current trial
            group_label = trial.group_label

            # Use regex to parse the group_label
            pattern = r"cohort='(.*?)', participant_id='(.*?)', time_measure='(.*?)', test='(.*?)', trial='(.*?)'"
            match = re.search(pattern, str(group_label))

            if match:                
                # Construct the tuple for MultiIndex access
                group_tuple = (
                    match.group(1),  # cohort
                    match.group(2),  # participant_id
                    match.group(3),  # time_measure
                    match.group(4),  # test
                    match.group(5),  # trial
                )

                # Check if the group_tuple is in gs_matches index
                if group_tuple in gs_matches.index:
                    print(f"Processing group_label: {group_label}")
                    matched_ref[group_label] = all_ref_parameters.loc[group_tuple]
                    matched_det[group_label] = all_per_wb_parameters.loc[group_tuple]

                    # Classify samples as tp, fp, tn, fn
                    n_samples = len(trial.data_ss)
                    detected_intervals = matched_det[group_label]
                    display(detected_intervals)
                    reference_intervals = matched_ref[group_label].drop(columns=["termination_reason"])
                    display(reference_intervals)

                    sample_wise_classifications = classify_samples(n_samples, detected_intervals, reference_intervals)

                    # Calculate sample-wise metrics
                    tp = (sample_wise_classifications == 'tp').sum()
                    fp = (sample_wise_classifications == 'fp').sum()
                    fn = (sample_wise_classifications == 'fn').sum()
                    tn = (sample_wise_classifications == 'tn').sum()

                    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0

                    trial_metrics = {
                        "trial": trial.group_label,
                        "accuracy": accuracy,
                        "sensitivity": sensitivity,
                        "specificity": specificity,
                        "ppv": ppv,
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "tn": tn
                    }
                    sample_wise_metrics_per_trial.append(trial_metrics)
                else:
                    print(f"Group label tuple {group_tuple} not found in all_ref_parameters.")
        
        print("Sample-wise Metrics per Trial:")
        display(pd.DataFrame(sample_wise_metrics_per_trial))
        
        # Aggregate sample-wise metrics per subject
        sample_wise_metrics_df = pd.DataFrame(sample_wise_metrics_per_trial)
        aggregated_sample_wise_metrics_subject = sample_wise_metrics_df[["accuracy", "sensitivity", "specificity", "ppv"]].mean().to_dict()
        aggregated_sample_wise_metrics_subject.update({
            "participant": participant_folder,
            "tp": sample_wise_metrics_df["tp"].sum(),
            "fp": sample_wise_metrics_df["fp"].sum(),
            "fn": sample_wise_metrics_df["fn"].sum(),
            "tn": sample_wise_metrics_df["tn"].sum()
        })

        columns_sw = ["participant", "accuracy", "sensitivity", "specificity", "ppv", "tp", "fp", "fn", "tn"]

        # Convert to a DataFrame
        aggregated_sample_wise_metrics_subject_df = pd.DataFrame([aggregated_sample_wise_metrics_subject], columns=columns_sw).reset_index(drop=True)
        aggregated_sample_wise_metrics.append(aggregated_sample_wise_metrics_subject_df)
        print(f'Aggregated Sample-wise Metrics with overlap th of {overlap_th}:')
        display(aggregated_sample_wise_metrics_subject_df)


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
            filenames=[f"ws_with_errors_{participant_folder}_th{overlap_th}.csv", f"ws_agg_results_{participant_folder}_th{overlap_th}.csv"]
        )

        print('-----------------------------------')
        print('\n')

#%% Run if you want to process all participants
global_results = pd.concat(all_subj_with_errors)
print('Global walking speed with errors:')
display(global_results.head())

# Calculate agg_results for the global_combined_tp_with_errors
global_aggregation = [
    *[(("walking_speed_mps", o), ["mean", A.quantiles]) for o in ["detected", "reference","abs_error", "abs_rel_error"]],
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
print('Global aggregated results:')
display(global_agg_results_truncated)

global_sw_results = pd.concat(aggregated_sample_wise_metrics)

# Calculate totals for count columns and means for metric columns
totals = global_sw_results[['tp', 'fp', 'fn', 'tn']].sum()
means = global_sw_results[['accuracy', 'sensitivity', 'specificity', 'ppv']].mean().map(truncate_to_decimals)

# Combine totals and means into a summary row
summary_data = {
    'participant': 'Summary',
    'accuracy': [means['accuracy'], truncate_to_decimals(A.loa(global_sw_results['accuracy']))],
    'sensitivity': [means['sensitivity'], truncate_to_decimals(A.loa(global_sw_results['sensitivity']))],
    'specificity': [means['specificity'], truncate_to_decimals(A.loa(global_sw_results['specificity']))],
    'ppv': [means['ppv'], truncate_to_decimals(A.loa(global_sw_results['ppv']))],
    'tp': totals['tp'],
    'fp': totals['fp'],
    'fn': totals['fn'],
    'tn': totals['tn']
}

# Wrap the summary data in a list to create a single-row DataFrame
summary_row = pd.DataFrame([summary_data])

# Append the summary row to the original DataFrame
global_sw_results_with_summary = pd.concat([global_sw_results, summary_row], ignore_index=True)
print(f'Global Sample-wise Metrics with overlap th of {overlap_th}:')
display(global_sw_results_with_summary)

# Save the global results
results_path = os.path.join(base_dir, "CohortResults")
os.makedirs(results_path, exist_ok=True)
global_results.to_csv(os.path.join(results_path, f'inLab_ws_with_errors_th{overlap_th}.csv'))
global_agg_results_truncated.to_csv(os.path.join(results_path, f"inLab_ws_agg_results_th{overlap_th}.csv"))
# Save aggregated sample-wise metrics for all subjects
all_sample_wise_path = os.path.join(results_path, f'inLab_gsd_fullpipeline_aggregated_sample_wise_metrics_th{overlap_th}.csv')
global_sw_results_with_summary.to_csv(all_sample_wise_path, index=False)

# %%
