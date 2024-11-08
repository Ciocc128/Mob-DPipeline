# %%
import os
import pandas as pd
from mobgap.data import GenericMobilisedDataset
from mobgap.gait_sequences import GsdAdaptiveIonescu
from mobgap.utils.df_operations import create_multi_groupby
from IPython.display import display

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def classify_samples(n_samples, detected_intervals, reference_intervals):
    classifications = pd.Series(['tn'] * n_samples)

    for _, row in detected_intervals.iterrows():
        for i in range(row['start'], row['end'] + 1):
            if classifications[i] == 'tn':
                classifications[i] = 'fp'
            elif classifications[i] == 'fn':
                classifications[i] = 'tp'

    for _, row in reference_intervals.iterrows():
        for i in range(row['start'], row['end'] + 1):
            if classifications[i] == 'tn':
                classifications[i] = 'fn'
            elif classifications[i] == 'fp':
                classifications[i] = 'tp'

    return classifications

def calculate_metrics(classifications):
    tp = (classifications == 'tp').sum()
    fp = (classifications == 'fp').sum()
    fn = (classifications == 'fn').sum()
    tn = (classifications == 'tn').sum()

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
    }

# Function to process a single participant
def process_participant(participant_id, data_path):
    mobDataset = GenericMobilisedDataset(
        [os.path.join(data_path, "data.mat")],
        test_level_names=["time_measure", "test", "trial"],
        reference_system='INDIP',
        measurement_condition='laboratory',
        reference_para_level='wb',
        parent_folders_as_metadata=["cohort", "participant_id"]
    )

    participant_metrics = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    for trial in mobDataset[3:]:
        imu_data = trial.data_ss
        reference_data = trial.reference_parameters_.wb_list
        gs_output = GsdAdaptiveIonescu().detect(imu_data, sampling_rate_hz=trial.sampling_rate_hz)

        detected_intervals = gs_output.gs_list_
        reference_intervals = reference_data

        n_samples = len(imu_data)
        classifications = classify_samples(n_samples, detected_intervals, reference_intervals)

        trial_metrics = calculate_metrics(classifications)
        
        # Aggregate metrics across all trials for the participant
        participant_metrics["tp"] += trial_metrics["tp"]
        participant_metrics["fp"] += trial_metrics["fp"]
        participant_metrics["fn"] += trial_metrics["fn"]
        participant_metrics["tn"] += trial_metrics["tn"]

    # Calculate participant-level metrics
    total_tp = participant_metrics["tp"]
    total_fp = participant_metrics["fp"]
    total_fn = participant_metrics["fn"]
    total_tn = participant_metrics["tn"]

    accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn) if (total_tp + total_tn + total_fp + total_fn) > 0 else 0
    sensitivity = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    specificity = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0
    ppv = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0

    participant_summary = {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "tp_samples": total_tp,
        "fp_samples": total_fp,
        "fn_samples": total_fn,
        "tn_samples": total_tn
    }

    return participant_summary

# Main function to process all participants
def process_all_participants(base_path):
    all_participant_results = []
    
    for participant_folder in os.listdir(base_path):
        participant_path = os.path.join(base_path, participant_folder)
        
        if os.path.isdir(participant_path) and "data.mat" in os.listdir(participant_path):
            print(f"Processing participant: {participant_folder}")
            
            participant_metrics = process_participant(participant_folder, participant_path)
            participant_metrics["participant_id"] = participant_folder
            
            all_participant_results.append(participant_metrics)
            display(pd.Series(participant_metrics))

    # Convert results to DataFrame and save to CSV
    all_results_df = pd.DataFrame(all_participant_results)
    all_results_df_path = os.path.join(base_path, "all_participants_gsd_metrics.csv")
    all_results_df.to_csv(all_results_df_path, index=False)
    print(f"All participants' metrics saved to {all_results_df_path}")

# Define the base directory containing participant folders
base_dir = 'C:/Users/ac4gt/Desktop/Mob-DPipeline/smartphone/test_data/lab/HA/'
# %%
# Run the processing for all participants
process_all_participants(base_dir)

# %%
import os
import pandas as pd
import matplotlib.pyplot as plt
from mobgap.data import GenericMobilisedDataset
from mobgap.gait_sequences import GsdAdaptiveIonescu
from mobgap.utils.df_operations import create_multi_groupby
from mobgap.pipeline.evaluation import categorize_intervals, get_matching_intervals
from IPython.display import display

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Helper function to classify samples
def classify_samples(n_samples, detected_intervals, reference_intervals):
    classifications = pd.Series(['tn'] * n_samples)

    for _, row in detected_intervals.iterrows():
        for i in range(row['start'], row['end'] + 1):
            if classifications[i] == 'tn':
                classifications[i] = 'fp'
            elif classifications[i] == 'fn':
                classifications[i] = 'tp'

    for _, row in reference_intervals.iterrows():
        for i in range(row['start'], row['end'] + 1):
            if classifications[i] == 'tn':
                classifications[i] = 'fn'
            elif classifications[i] == 'fp':
                classifications[i] = 'tp'

    return classifications

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
    sample_wise_metrics_per_trial = []

    # Process trials from the 4th one onward
    for trial in mobDataset[3:]:
        imu_data = trial.data_ss
        reference_data = trial.reference_parameters_.wb_list
        gs_output = GsdAdaptiveIonescu().detect(imu_data, sampling_rate_hz=trial.sampling_rate_hz)

        # Save reference parameters for later comparison
        all_ref_parameters[trial.group_label] = reference_data
        per_wb_paras[trial.group_label] = gs_output.gs_list_

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

    return per_wb_parameters_df, ref_parameters_df, sample_wise_metrics_per_trial

# Main function to process all participants
def process_all_participants(base_path, plot_gsd=False):
    # Iterate over all participant folders
    for participant_folder in os.listdir(base_path):
        participant_path = os.path.join(base_path, participant_folder)
        
        if os.path.isdir(participant_path) and "data.mat" in os.listdir(participant_path):
            print(f"Processing participant: {participant_folder}")
            
            # Process the participant data
            per_wb_parameters, ref_parameters, sample_wise_metrics_per_trial = process_participant(
                participant_folder, participant_path, plot_gsd=plot_gsd
            )

            # Direct matching method
            index_names = ["cohort", "participant_id", "time_measure", "test", "trial", "wb_id"]
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

            precision = tp_samples / (tp_samples + fp_samples) if (tp_samples + fp_samples) > 0 else 0
            recall = tp_samples / (tp_samples + fn_samples) if (tp_samples + fn_samples) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            direct_matching_metrics = {
                "tp_samples": tp_samples,
                "fp_samples": fp_samples,
                "fn_samples": fn_samples,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
            }

            print("Direct Matching Metrics:")
            display(pd.Series(direct_matching_metrics))

            # Aggregate sample-wise metrics per subject
            sample_wise_metrics_df = pd.DataFrame(sample_wise_metrics_per_trial)
            aggregated_sample_wise_metrics = sample_wise_metrics_df[["accuracy", "sensitivity", "specificity", "ppv"]].mean().to_dict()
            aggregated_sample_wise_metrics.update({
                "tp": sample_wise_metrics_df["tp"].sum(),
                "fp": sample_wise_metrics_df["fp"].sum(),
                "fn": sample_wise_metrics_df["fn"].sum(),
                "tn": sample_wise_metrics_df["tn"].sum()
            })

            print("Aggregated Sample-wise Metrics per Subject:")
            display(pd.Series(aggregated_sample_wise_metrics))

            # Save results to CSV
            results_folder = os.path.join(participant_path, "results")
            if not os.path.exists(results_folder):
                os.makedirs(results_folder)

            # Save direct matching metrics
            direct_matching_path = os.path.join(results_folder, f"direct_matching_metrics_{participant_folder}.csv")
            pd.DataFrame([direct_matching_metrics]).to_csv(direct_matching_path, index=False)

            # Save aggregated sample-wise metrics
            sample_wise_aggregated_path = os.path.join(results_folder, f"aggregated_sample_wise_metrics_{participant_folder}.csv")
            pd.DataFrame([aggregated_sample_wise_metrics]).to_csv(sample_wise_aggregated_path, index=False)

            print(f"Direct matching metrics saved to {direct_matching_path}")
            print(f"Aggregated sample-wise metrics saved to {sample_wise_aggregated_path}")

# Define the base directory containing participant folders
base_dir = 'C:/Users/ac4gt/Desktop/Mob-DPipeline/smartphone/test_data/lab/HA/'

# Run the processing for all participants
process_all_participants(base_dir, plot_gsd=False)


# %%
