#%%
from tqdm import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt
from mobgap.data import GenericMobilisedDataset
from mobgap.gait_sequences import GsdAdaptiveIonescu, GsdIluz
from mobgap.utils.conversions import to_body_frame
from mobgap.utils.df_operations import create_multi_groupby, apply_transformations, apply_aggregations, CustomOperation
from mobgap.pipeline.evaluation import categorize_intervals, get_matching_intervals, ErrorTransformFuncs as E
from mobgap.pipeline.evaluation import CustomErrorAggregations as A
from IPython.display import display


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Helper function to plot results with title
def plot_gsd_outputs(data, title, **kwargs):
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
    ax.set_title(title)  # Set the title with the trial information
    return fig, ax

# Helper function to classify samples
def classify_samples(n_samples, detected_intervals, reference_intervals):
    # Initialize all samples as 'tn' (true negative)
    classifications = pd.Series(['tn'] * n_samples)

    # Process detected intervals for tp and fp classifications
    for _, row in detected_intervals.iterrows():
        for i in range(row['start'], row['end'] + 1):
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
        for i in range(row['start'], row['end'] + 1):
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

def truncate_to_decimals(x):
    if isinstance(x, float):  # Applica solo ai numeri float
        return round(x, 3)
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

# Function to process a single participant with sample-wise method
def process_participant_sample_wise(participant_id, mobDataset, plot_gsd=False):
    """
    Process a single participant for sample-wise metrics.

    Args:
        participant_id (str): ID of the participant being processed.
        mobDataset (GenericMobilisedDataset): Pre-loaded dataset for the participant.
        plot_gsd (bool): Whether to plot the GSD outputs for each trial.
    
    Returns:
        per_wb_parameters_df (DataFrame): Walking bout parameters for the participant.
        ref_parameters_df (DataFrame): Reference parameters for the participant.
        sample_wise_metrics_per_trial (list): Sample-wise metrics for each trial.
    """
    per_wb_paras = {}
    all_ref_parameters = {}
    sample_wise_metrics_per_trial = []
    detector = "iluz"  # Set the detector to use

    # Process trials from the 4th one onward
    for trial in tqdm(mobDataset[3:]):
        imu_data = trial.data_ss
        reference_data = trial.reference_parameters_.wb_list
        if detector == "iluz":
            gs_output = GsdIluz().detect(
                to_body_frame(imu_data), 
                sampling_rate_hz=trial.sampling_rate_hz
            )
        elif detector == "ionescu":
            gs_output = GsdAdaptiveIonescu().detect(
                imu_data,
                sampling_rate_hz=trial.sampling_rate_hz,
            )
        # Save reference parameters for later comparison
        all_ref_parameters[trial.group_label] = reference_data
        per_wb_paras[trial.group_label] = gs_output.gs_list_

        # Plotting (if enabled)
        if plot_gsd:
            title = f"Participant: {participant_id}, Trial: {trial.group_label}"
            fig, ax = plot_gsd_outputs(
                trial.data_ss,
                title=title,  # Pass the trial title to the plot function
                reference=reference_data,
                detected=gs_output.gs_list_,
            )
            plt.show()

        # Sample-wise classification
        n_samples = len(imu_data)
        sample_wise_classifications = classify_samples(n_samples, gs_output.gs_list_, reference_data)

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

    # Concatenate reference parameters and per-walking bout parameters
    index_names = ["cohort", "participant_id", "time_measure", "test", "trial", "wb_id"]
    ref_parameters_df = pd.concat(all_ref_parameters, names=index_names)
    per_wb_parameters_df = pd.concat(per_wb_paras, names=index_names)

    # Add "duration_s" column to per_wb_parameters_df
    per_wb_parameters_df["duration_s"] = (per_wb_parameters_df["end"] - per_wb_parameters_df["start"] + 1) / 100

    return per_wb_parameters_df, ref_parameters_df, sample_wise_metrics_per_trial

#base_dir = 'C:/Users/ac4gt/Desktop/Mob-DPipeline/smartphone/test_data/lab/HA/'
base_dir = 'C:/PoliTO/Tesi/mobgap/smartphone/test_data/lab/HA/'

participant_folder = False
#participant_folder ='011'
index_names = ["cohort", "participant_id", "time_measure", "test", "trial", "wb_id"]
participants = [participant_folder] if participant_folder else os.listdir(base_dir)

aggregated_direct_matching_metrics = []
aggregated_sample_wise_metrics = []
all_durations = []

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

        # Sample-wise method
        per_wb_parameters, ref_parameters, sample_wise_metrics_per_trial, = process_participant_sample_wise(
            participant_folder, mobDataset, plot_gsd=True
        )

        # Direct matching method
        per_trial_participant_grouper = create_multi_groupby(
            per_wb_parameters,
            ref_parameters,
            groupby=index_names[:-1],
        )            

        overlap_th = 0.8  # Set overlap threshold
        gs_tp_fp_fn = per_trial_participant_grouper.apply(
            lambda det, ref: categorize_intervals(
                gsd_list_detected=det,
                gsd_list_reference=ref,
                overlap_threshold=overlap_th,
                multiindex_warning=False
            )
        )

        gs_matches = get_matching_intervals(
            metrics_detected=per_wb_parameters,
            metrics_reference=ref_parameters,
            matches=gs_tp_fp_fn,
        )
        display(gs_matches.drop(columns=["wb_id"]))
        
        error_config = [
            ("duration_s", [E.abs_error]),   # Duration can have absolute error in seconds.
        ]

        errors = apply_transformations(gs_matches, error_config)
        gs_matches_with_errors = pd.concat([gs_matches, errors], axis=1).sort_index(axis=1)
        all_durations.append(gs_matches_with_errors.drop(columns=["wb_id"]))
        display(gs_matches_with_errors)

        # Aggregation
        aggregation = [
            *[(("duration_s", o), ["mean", A.quantiles]) for o in ["detected", "abs_error"]],
            *[CustomOperation(
                identifier="duration_s",
                function=A.icc,
                column_name=("duration_s", "all"),
            )],
            CustomOperation(identifier=None, function=A.n_datapoints, column_name=("all", "all")),
        ]
        agg_results = (
            apply_aggregations(gs_matches_with_errors.dropna(), aggregation)
            .rename_axis(index=["aggregation", "metric", "origin"])
            .reorder_levels(["metric", "origin", "aggregation"])
            .sort_index(level=0)
            .to_frame("values")
        )

        agg_results_truncated = agg_results.map(truncate_to_decimals)
        display(agg_results_truncated)

        # Count true positives, false positives, and false negatives
        tp_samples = len(gs_tp_fp_fn[gs_tp_fp_fn["match_type"] == "tp"])
        fp_samples = len(gs_tp_fp_fn[gs_tp_fp_fn["match_type"] == "fp"])
        fn_samples = len(gs_tp_fp_fn[gs_tp_fp_fn["match_type"] == "fn"])

        precision = tp_samples / (tp_samples + fp_samples) if (tp_samples + fp_samples) > 0 else 0
        recall = tp_samples / (tp_samples + fn_samples) if (tp_samples + fn_samples) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        direct_matching_metrics = {
            "participant": participant_folder,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "tp": tp_samples,
            "fp": fp_samples,
            "fn": fn_samples
        }

        # Display and append to aggregated list
        direct_matching_metrics_df = pd.DataFrame([direct_matching_metrics])
        print("Direct Matching Metrics:")
        display(direct_matching_metrics_df)
        aggregated_direct_matching_metrics.append(direct_matching_metrics_df)

        # Display sample-wise metrics for each trial
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
        print("Aggregated Sample-wise Metrics:")
        display(aggregated_sample_wise_metrics_subject_df)


        # Save results for the participant
        save_results_to_folder(
            dataframes=[gs_matches_with_errors, agg_results_truncated],
            participant_folder=participant_path,
            filenames=[f"gsd_with_errors_{participant_folder}.csv", f"gsd_agg_results_{participant_folder}.csv"]
        )

        # Save individual results to "results" folder within the participant folder
        results_folder = os.path.join(participant_path, "results")
        os.makedirs(results_folder, exist_ok=True)

        # Save direct matching metrics
        direct_matching_path = os.path.join(results_folder, f"gsd_direct_matching_metrics_{participant_folder}.csv")
        dataset_dm = pd.DataFrame([direct_matching_metrics])
        dataset_dm.to_csv(direct_matching_path, index=False)

        # Save sample-wise metrics per trial
        sample_wise_metrics_path = os.path.join(results_folder, f"gsd_sample_wise_metrics_per_trial_{participant_folder}.csv")
        pd.DataFrame(sample_wise_metrics_per_trial).to_csv(sample_wise_metrics_path, index=False)

        # Save aggregated sample-wise metrics
        aggregated_sample_wise_path = os.path.join(results_folder, f"gsd_aggregated_sample_wise_metrics_{participant_folder}.csv")
        dataset_sw = aggregated_sample_wise_metrics_subject_df
        dataset_sw.to_csv(aggregated_sample_wise_path, index=False)

        print(f"Results saved to {results_folder}")
        print("----------------------------------------")
        print("\n")

#%% Run only if you want to aggregate results for all participants
# Aggregate and save all subjects' metrics to "Allresults" folder
global_results = pd.concat(all_durations)
print("Global Results:")
display(global_results)

# Calculate agg_results for the global_combined_gsd_with_errors
global_aggregation = [
    *[(("duration_s", o), ["mean", A.quantiles]) for o in ["detected", "abs_error"]],
    *[CustomOperation(identifier="duration_s", function=A.icc, column_name=("duration_s", "all"))],
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
print("Aggregated Results:")
display(global_agg_results_truncated)

global_dm_results = pd.concat(aggregated_direct_matching_metrics)

totals = global_dm_results[['tp', 'fp', 'fn']].sum()
means = global_dm_results[['precision', 'recall', 'f1_score']].mean()

# Combine totals and means into a summary row
summary_row = pd.DataFrame(
    [
        {
        'participant': 'Summary',
        'precision': means['precision'],
        'recall': means['recall'],
        'f1_score': means['f1_score'],
        'tp': totals['tp'],
        'fp': totals['fp'],
        'fn': totals['fn']
        }
    ]
)

# Append the summary row to the DataFrame
global_dm_metrics_with_summary = pd.concat([global_dm_results, summary_row], ignore_index=True)
print("Global Direct Matching Metrics:")
display(global_dm_metrics_with_summary)

global_sw_results = pd.concat(aggregated_sample_wise_metrics)

# Calculate totals for count columns and means for metric columns
totals = global_sw_results[['tp', 'fp', 'fn', 'tn']].sum()
means = global_sw_results[['accuracy', 'sensitivity', 'specificity', 'ppv']].mean()

# Combine totals and means into a summary row
summary_data = {
    'participant': 'Summary',
    'accuracy': means['accuracy'],
    'sensitivity': means['sensitivity'],
    'specificity': means['specificity'],
    'ppv': means['ppv'],
    'tp': totals['tp'],
    'fp': totals['fp'],
    'fn': totals['fn'],
    'tn': totals['tn']
}

# Wrap the summary data in a list to create a single-row DataFrame
summary_row = pd.DataFrame([summary_data])

# Append the summary row to the original DataFrame
global_sw_results_with_summary = pd.concat([global_sw_results, summary_row], ignore_index=True)
print("Global Sample-wise Metrics:")
display(global_sw_results_with_summary)


all_results_folder = os.path.join(base_dir, "CohortResults")
os.makedirs(all_results_folder, exist_ok=True)

# Save the global results
global_results.to_csv(os.path.join(all_results_folder, "inLab_gsd_with_errors.csv"))
global_agg_results_truncated.to_csv(os.path.join(all_results_folder, "inLab_gsd_agg_results.csv"))

# Save aggregated direct matching metrics for all subjects
all_direct_matching_path = os.path.join(all_results_folder, "inLab_gsd_aggregated_direct_matching_metrics.csv")
global_dm_metrics_with_summary.to_csv(all_direct_matching_path, index=False)

# Save aggregated sample-wise metrics for all subjects
all_sample_wise_path = os.path.join(all_results_folder, "inLab_gsd_aggregated_sample_wise_metrics.csv")
global_sw_results_with_summary.to_csv(all_sample_wise_path, index=False)

print(f"Cohort results saved to {all_results_folder}")


# %%
