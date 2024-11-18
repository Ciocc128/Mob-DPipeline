#%%
import os
import pandas as pd
import matplotlib.pyplot as plt
from mobgap.data import GenericMobilisedDataset
from mobgap.initial_contacts import IcdShinImproved
from mobgap.utils.df_operations import apply_transformations, create_multi_groupby
from mobgap.pipeline.evaluation import ErrorTransformFuncs as E
from mobgap.utils.df_operations import apply_aggregations, CustomOperation
from mobgap.pipeline.evaluation import CustomErrorAggregations as A
from IPython.display import display
from mobgap.utils.conversions import to_body_frame, as_samples
from mobgap.pipeline import GsIterator
from mobgap.initial_contacts.evaluation import categorize_ic_list, calculate_matched_icd_performance_metrics
import numpy as np

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Helper function to save DataFrames
def save_results_to_folder(dataframes, participant_folder, filenames):
    results_dir = os.path.join(participant_folder, "results")
    os.makedirs(results_dir, exist_ok=True)  # Create the results directory if it doesn't exist
    
    for df, filename in zip(dataframes, filenames):
        file_path = os.path.join(results_dir, filename)
        df.to_csv(file_path, index=True)  # Save with index for traceability
        print(f"Saved {filename} to {file_path}")


# Helper function to calculate ICD Shin output for a single trial
def calculate_icd_shin_output(single_test_data, plot):
    imu_data = to_body_frame(single_test_data.data_ss)
    sampling_rate_hz = single_test_data.sampling_rate_hz
    reference_wbs = single_test_data.reference_parameters_.wb_list

    # Initialize iterator for processing
    iterator = GsIterator()
    final_filtered_signal = None  # To store the filtered signal for plotting
    ic_list_internal = []  # To store detected ICs for plotting

    for (gs, data), result in iterator.iterate(imu_data, reference_wbs):
        icd_shin = IcdShinImproved()
        detected = icd_shin.detect(data, sampling_rate_hz=sampling_rate_hz)
        result.ic_list = detected.ic_list_
        final_filtered_signal = icd_shin.final_filtered_signal_  # Store filtered signal
        ic_list_internal = icd_shin.ic_list_internal_  # Internal ICs detected for plotting

    # Convert ic_list_internal to integer indices
    ic_list_internal = np.array(ic_list_internal).astype(int)  # Convert to integer array

    # Detected ICs from iterator
    det_ics = iterator.results_.ic_list

    if plot:
        # Plot filtered signal and detected ICs
        plt.figure(figsize=(12, 6))
        plt.plot(final_filtered_signal, label='Filtered Signal')  # Plot filtered signal
        plt.scatter(ic_list_internal, final_filtered_signal[ic_list_internal], color='red', label='Detected ICs')  # Mark detected ICs
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude')
        plt.title('Filtered Signal with Detected Initial Contacts (ICs)')
        plt.legend()
        plt.show()

    return det_ics, imu_data

base_dir = 'C:/Users/ac4gt/Desktop/Mob-DPipeline/smartphone/test_data/lab/HA/'
participant_folder = False
plot = False
index_names = ["cohort", "participant_id", "time_measure", "test", "trial"]

participants = [participant_folder] if participant_folder else os.listdir(base_dir)
# Initialize an empty list to store combined_tp_with_errors from all participants
all_combined_tp_with_errors = []
all_metrics_per_wb = []

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

        all_combined_tp = {}
        participant_metrics_per_wb = {}

        for trial in mobDataset[3:]:  # Process trials from the 4th one onward skipping standing and data personalization
            detected_ics, imu_data = calculate_icd_shin_output(trial, plot=False)
            reference_ics = trial.reference_parameters_.ic_list

            # Calculate matches_per_wb and metrics_per_wb for each trial
            tolerance_s = 0.2
            sampling_rate_hz = 100  # Adjust according to your data
            tolerance_samples = as_samples(tolerance_s, sampling_rate_hz)
            per_wb_grouper = create_multi_groupby(detected_ics, reference_ics, groupby="wb_id")

            # Calculate matches per wb
            matches_per_wb = per_wb_grouper.apply(
                lambda df1, df2: categorize_ic_list(
                    ic_list_detected=df1,
                    ic_list_reference=df2,
                    tolerance_samples=tolerance_samples,
                    multiindex_warning=False,
                )
            )

            # Calculate metrics per wb
            metrics_per_wb = matches_per_wb.groupby(level="wb_id").apply(
                lambda df_: pd.Series(calculate_matched_icd_performance_metrics(df_))
            )

            # Store trial-level metrics
            participant_metrics_per_wb[trial.group_label] = metrics_per_wb

            tp_matches = matches_per_wb[matches_per_wb['match_type'] == 'tp']

            detected_tp_ics = []
            reference_tp_ics = []
            for index, row in tp_matches.iterrows():
                wb_id = index[0]
                detected_idx = row['ic_id_detected'][1]
                reference_idx = row['ic_id_reference'][1]

                detected_tp_ics.append(detected_ics.loc[(wb_id, detected_idx), 'ic'])
                reference_tp_ics.append(reference_ics.loc[(wb_id, reference_idx), 'ic'])
            
            combined_tp = pd.DataFrame({
                'detected': detected_tp_ics,
                'reference': reference_tp_ics
            })

            combined_tp.columns = pd.MultiIndex.from_product([['ic'], combined_tp.columns])
            all_combined_tp[trial.group_label] = combined_tp

        # Concatenate per-participant results
        combined_tp_df = pd.concat(all_combined_tp, names=index_names)
        metrics_per_wb_df = pd.concat(participant_metrics_per_wb, names=index_names)

        errors = [("ic", [E.abs_error, E.rel_error])]
        errors = apply_transformations(combined_tp_df, errors)

        combined_tp_with_errors = pd.concat([combined_tp_df, errors], axis=1).sort_index(axis=1)
        multiindex_column_order = [
            ('ic', 'detected'), ('ic', 'reference'), ('ic', 'abs_error'), ('ic', 'rel_error')
        ]
        combined_tp_with_errors = combined_tp_with_errors.reindex(columns=multiindex_column_order)

        # Append to global lists
        all_combined_tp_with_errors.append(combined_tp_with_errors)
        all_metrics_per_wb.append(metrics_per_wb_df)

        # Save results for the participant
        save_results_to_folder(
            dataframes=[combined_tp_with_errors, metrics_per_wb_df],
            participant_folder=participant_path,
            filenames=[f"icd_tp_with_errors_{participant_folder}.csv", f"metrics_per_wb_{participant_folder}.csv"]
        )

# Concatenate all participants' combined_tp_with_errors and metrics_per_wb into single DataFrames
global_combined_tp_with_errors = pd.concat(all_combined_tp_with_errors)
global_metrics_per_wb = pd.concat(all_metrics_per_wb)

# Calculate totals and means
totals = global_metrics_per_wb[['tp_samples', 'fp_samples', 'fn_samples']].sum()
means = global_metrics_per_wb[['precision', 'recall', 'f1_score']].mean()

# Combine totals and means into a summary row
summary_row = pd.DataFrame(
    {
        'tp_samples': [totals['tp_samples']],
        'fp_samples': [totals['fp_samples']],
        'fn_samples': [totals['fn_samples']],
        'precision': [means['precision']],
        'recall': [means['recall']],
        'f1_score': [means['f1_score']]
    },
    index=[('Summary', 'Total', 'Mean', '', '')]  # MultiIndex for the summary row
)

# Ensure the summary row matches the DataFrame's column structure
summary_row.columns = global_metrics_per_wb.columns

# Append the summary row to the DataFrame
global_metrics_with_summary = pd.concat([global_metrics_per_wb, summary_row])

# Save the concatenated DataFrames
results_path = os.path.join(base_dir, "CohortResults")
os.makedirs(results_path, exist_ok=True)
global_combined_tp_with_errors.to_csv(os.path.join(results_path, "icd_tp_with_errors.csv"), index=True)
global_metrics_with_summary.to_csv(os.path.join(results_path, "icd_metrics_per_wb.csv"), index=True)

# Calculate agg_results for the global_combined_tp_with_errors
global_aggregation = [
    *[(("ic", o), ["mean", "std"]) for o in ["abs_error", "rel_error"]],
    CustomOperation(identifier="ic", function=A.icc, column_name=("ic", "all"))
]
global_agg_results = (
    apply_aggregations(global_combined_tp_with_errors.dropna(), global_aggregation)
    .rename_axis(index=["aggregation", "metric", "origin"])
    .reorder_levels(["metric", "origin", "aggregation"])
    .sort_index(level=0)
    .to_frame("values")
)

# Save the aggregated results
global_agg_results.to_csv(os.path.join(results_path, "icd_agg_results.csv"), index=True)

print("Global results saved:")
print(f"Combined TP with Errors: {os.path.join(results_path, 'icd_tp_with_errors.csv')}")
print(f"Metrics per WB: {os.path.join(results_path, 'icd_metrics_per_wb.csv')}")
print(f"Aggregated Results: {os.path.join(results_path, 'icd_agg_results.csv')}")
