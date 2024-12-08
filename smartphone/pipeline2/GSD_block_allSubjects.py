#%%  Direct Matching WBs
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
from mobgap.utils.evaluation import precision_recall_f1_score

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

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

# Function to process a single participant
def process_participant(participant_id, data_path, plot_gsd=False):
    # Load the dataset for the participant
    mobDataset = GenericMobilisedDataset(
        [os.path.join(data_path, "data.mat")],
        test_level_names=["time_measure", "test", "trial"],
        reference_system='INDIP',
        measurement_condition='laboratory',
        reference_para_level='wb',
        parent_folders_as_metadata=["cohort", "participant_id"]
    )

    per_wb_paras = {}
    all_ref_parameters = {}

    # Process trials from the 4th one onward
    for trial in mobDataset[3:]:
        imu_data = trial.data_ss
        reference_data = trial.reference_parameters_.wb_list
        gs_output = GsdAdaptiveIonescu().detect(imu_data, sampling_rate_hz=trial.sampling_rate_hz)

        # Save reference parameters for later comparison
        all_ref_parameters[trial.group_label] = reference_data
        per_wb_paras[trial.group_label] = gs_output.gs_list_

        # Plotting (if enabled)
        if plot_gsd:
            fig, ax = plot_gsd_outputs(
                trial.data_ss,
                reference=reference_data,
                detected=gs_output.gs_list_,
            )
            plt.show()

    # Concatenate reference parameters and per-walking bout parameters
    index_names = ["cohort", "participant_id", "time_measure", "test", "trial", "wb_id"]
    ref_parameters_df = pd.concat(all_ref_parameters, names=index_names)
    per_wb_parameters_df = pd.concat(per_wb_paras, names=index_names)

    return per_wb_parameters_df, ref_parameters_df

# Main function to process all participants
# Main function to process all participants
def process_all_participants(base_path, plot_gsd=False):
    
    # Iterate over all participant folders
    for participant_folder in os.listdir(base_path):
        participant_path = os.path.join(base_path, participant_folder)
        
        if os.path.isdir(participant_path) and "data.mat" in os.listdir(participant_path):
            print(f"Processing participant: {participant_folder}")
            
            # Process the participant data
            per_wb_parameters, ref_parameters = process_participant(participant_folder, participant_path, plot_gsd=plot_gsd)

            # Compare detected walking bouts with reference
            index_names = ["cohort","participant_id","time_measure", "test", "trial", "wb_id"]
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
            display(gs_tp_fp_fn)

            # Count true positives, false positives, and false negatives
            tp_samples = len(gs_tp_fp_fn[gs_tp_fp_fn["match_type"] == "tp"])
            fp_samples = len(gs_tp_fp_fn[gs_tp_fp_fn["match_type"] == "fp"])
            fn_samples = len(gs_tp_fp_fn[gs_tp_fp_fn["match_type"] == "fn"])

            # Manually calculate precision, recall, and F1-score
            precision = tp_samples / (tp_samples + fp_samples) if (tp_samples + fp_samples) > 0 else 0
            recall = tp_samples / (tp_samples + fn_samples) if (tp_samples + fn_samples) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            # Compile metrics dictionary
            gsd_metrics = {
                "tp_samples": tp_samples,
                "fp_samples": fp_samples,
                "fn_samples": fn_samples,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
            }

            print("GSD Performance Metrics:")
            display(pd.Series(gsd_metrics))

            # Get matching intervals for error calculations
            gs_matches = get_matching_intervals(
                metrics_detected=per_wb_parameters,
                metrics_reference=ref_parameters,
                matches=gs_tp_fp_fn,
            )
            display(gs_matches)

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

            # Apply aggregation focusing on 'start' and 'end' parameters
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

            # Save results to CSV
            results_folder = os.path.join(participant_path, "results")
            if not os.path.exists(results_folder):
                os.makedirs(results_folder)

            gs_errors_path = os.path.join(results_folder, f"gsd_errors_subject_{participant_folder}.csv")
            gsd_metrics_path = os.path.join(results_folder, f"gsd_metrics_subject_{participant_folder}.csv")
            agg_results_path = os.path.join(results_folder, f"gsd_agg_results_subject_{participant_folder}.csv")

            # Save the metrics and error results
            display(gs_matches_with_errors)
            gs_matches_with_errors.to_csv(gs_errors_path)
            display(agg_results)
            agg_results.to_csv(agg_results_path)
            
            gsd_metrics_df = pd.DataFrame([gsd_metrics])
            gsd_metrics_df.to_csv(gsd_metrics_path, index=False)

            print(f"GS matches with errors saved to {gs_errors_path}")
            print(f"Aggregated results saved to {agg_results_path}")
            print(f"Performance metrics saved to {gsd_metrics_path}")

# Define the base directory containing participant folders
base_dir = 'C:/Users/ac4gt/Desktop/Mob-DPipeline/smartphone/test_data/lab/HA/'

#%% Run the processing for all participants
process_all_participants(base_dir, plot_gsd=False)




# %% Sample Wise method
import os
import pandas as pd
from mobgap.data import GenericMobilisedDataset
from mobgap.gait_sequences import GsdAdaptiveIonescu
from mobgap.utils.df_operations import create_multi_groupby
from mobgap.gait_sequences.evaluation import categorize_intervals_per_sample, calculate_matched_gsd_performance_metrics
from IPython.display import display

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Helper function to plot results (optional, not used here)
def plot_gsd_outputs(data, **kwargs):
    pass

# Function to process a single participant
def process_participant(participant_id, data_path, plot_gsd=False):
    # Load the dataset for the participant
    mobDataset = GenericMobilisedDataset(
        [os.path.join(data_path, "data.mat")],
        test_level_names=["time_measure", "test", "trial"],
        reference_system='INDIP',
        measurement_condition='laboratory',
        reference_para_level='wb',
        parent_folders_as_metadata=["cohort", "participant_id"]
    )

    per_wb_paras = {}
    all_ref_parameters = {}

    # Process trials from the 4th one onward
    for trial in mobDataset[3:]:
        imu_data = trial.data_ss
        reference_data = trial.reference_parameters_.wb_list
        gs_output = GsdAdaptiveIonescu().detect(imu_data, sampling_rate_hz=trial.sampling_rate_hz)

        # Save reference parameters for later comparison
        all_ref_parameters[trial.group_label] = reference_data
        per_wb_paras[trial.group_label] = gs_output.gs_list_

        # Use the total length of the imu_data as n_overall_samples
        n_overall_samples = len(imu_data)

        # Debug print statements to trace the issue
        max_end_detected = gs_output.gs_list_["end"].max()
        max_end_reference = reference_data["end"].max()
        print(f"Participant: {participant_id}, Trial: {trial.group_label}")
        print(f"n_overall_samples: {n_overall_samples}, Max end detected: {max_end_detected}, Max end reference: {max_end_reference}")

        # Check if n_overall_samples is large enough
        if n_overall_samples <= max(max_end_detected, max_end_reference):
            print(f"Warning: n_overall_samples is too small for Trial {trial.group_label} in Participant {participant_id}")

    # Concatenate reference parameters and per-walking bout parameters
    index_names = ["cohort", "participant_id", "time_measure", "test", "trial", "wb_id"]
    ref_parameters_df = pd.concat(all_ref_parameters, names=index_names)
    per_wb_parameters_df = pd.concat(per_wb_paras, names=index_names)

    return per_wb_parameters_df, ref_parameters_df, n_overall_samples

# Function to process each group for categorization and metrics calculation
def process_gait_sequence_group(det, ref, n_overall_samples):
    # Print det and ref for debugging
    print(f"\nDetected GSD List (det):\n{det}")
    print(f"Reference GSD List (ref):\n{ref}")
    
    # Categorize intervals per sample
    categorized_intervals = categorize_intervals_per_sample(
        gsd_list_detected=det,
        gsd_list_reference=ref,
        n_overall_samples=n_overall_samples
    )
    return categorized_intervals

# Main function to process all participants
def process_all_participants(base_path, plot_gsd=False):
    # Iterate over all participant folders
    for participant_folder in os.listdir(base_path):
        participant_path = os.path.join(base_path, participant_folder)
        
        if os.path.isdir(participant_path) and "data.mat" in os.listdir(participant_path):
            print(f"Processing participant: {participant_folder}")
            
            # Process the participant data
            per_wb_parameters, ref_parameters, n_overall_samples = process_participant(participant_folder, participant_path, plot_gsd=plot_gsd)
            print(f"Total samples in the recording: {n_overall_samples}")

            # Compare detected walking bouts with reference
            index_names = ["time_measure", "test", "trial", "wb_id"]
            per_trial_participant_grouper = create_multi_groupby(
                per_wb_parameters,
                ref_parameters,
                groupby=index_names[:-1],
            )

            # Sample-wise method
            try:
                all_categorized_intervals = []
                for name, (det, ref) in per_trial_participant_grouper:
                    print(f"\nProcessing group: {name}")
                    categorized_intervals = process_gait_sequence_group(det, ref, n_overall_samples)
                    all_categorized_intervals.append(categorized_intervals)
                
                # Concatenate all categorized intervals into a single DataFrame
                gs_tp_fp_fn = pd.concat(all_categorized_intervals, ignore_index=True)
                display(gs_tp_fp_fn)

                # Calculate performance metrics using matched samples, including TN
                metrics = calculate_matched_gsd_performance_metrics(gs_tp_fp_fn)

                # Calculate PPV manually
                tp = len(gs_tp_fp_fn[gs_tp_fp_fn["match_type"] == "tp"])
                fp = len(gs_tp_fp_fn[gs_tp_fp_fn["match_type"] == "fp"])
                fn = len(gs_tp_fp_fn[gs_tp_fp_fn["match_type"] == "fn"])
                tn = len(gs_tp_fp_fn[gs_tp_fp_fn["match_type"] == "tn"])

                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
                accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

                # Compile metrics dictionary
                gsd_metrics = {
                    "tp_samples": tp,
                    "fp_samples": fp,
                    "fn_samples": fn,
                    "tn_samples": tn,
                    "accuracy": accuracy,
                    "sensitivity": metrics["recall"],  # Sensitivity or recall
                    "specificity": metrics["specificity"],
                    "ppv": ppv,
                }

                print("GSD Performance Metrics:")
                display(pd.Series(gsd_metrics))

                # Save results to CSV
                results_folder = os.path.join(participant_path, "results")
                if not os.path.exists(results_folder):
                    os.makedirs(results_folder)

                gsd_metrics_path = os.path.join(results_folder, f"gsd_metrics_subject_{participant_folder}.csv")

                gsd_metrics_df = pd.DataFrame([gsd_metrics])
                gsd_metrics_df.to_csv(gsd_metrics_path, index=False)

                print(f"Performance metrics saved to {gsd_metrics_path}")

            except ValueError as e:
                print(f"Error processing trial for participant {participant_folder}: {e}")

# Define the base directory containing participant folders
base_dir = 'C:/Users/ac4gt/Desktop/Mob-DPipeline/smartphone/test_data/lab/HA/'

# %% Run the processing for all participants
process_all_participants(base_dir, plot_gsd=False)


# %%
