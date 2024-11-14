# %%
import os
import pandas as pd
from mobgap.data import GenericMobilisedDataset
from mobgap.stride_length import SlZijlstra
from mobgap.pipeline import GsIterator
from mobgap.initial_contacts import refine_gs
from mobgap.utils.conversions import to_body_frame
from smartphone.pipeline2.riorientation.riorientamento import process_and_rotate_dataset
from IPython.display import display
import matplotlib.pyplot as plt
from mobgap.pipeline.evaluation import ErrorTransformFuncs as E
from mobgap.utils.df_operations import apply_transformations, apply_aggregations, CustomOperation
from mobgap.pipeline.evaluation import CustomErrorAggregations as A
from mobgap.data import LabExampleDataset

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def process_participant_data(participant_id, data_path):
    """Process data for a single participant and return stride lengths (detected and reference)."""
    
    # Load the dataset for the participant
    mobDataset = GenericMobilisedDataset(
        [os.path.join(data_path, "data.mat")],
        test_level_names=["time_measure", "test", "trial"],
        reference_system='INDIP',
        measurement_condition='laboratory',
        reference_para_level='wb',
        parent_folders_as_metadata=["cohort", "participant_id"]
    )
    
    all_detected_sl = {}
    all_ref_sl = {}

    sl_calculator = SlZijlstra(
        **SlZijlstra.PredefinedParameters.step_length_scaling_factor_ms_all
    )
    
    for trial in mobDataset[3:]:
        imu_data = trial.data_ss
        sampling_rate_hz = trial.sampling_rate_hz
        reference_wbs = trial.reference_parameters_.wb_list
        reference_ic = trial.reference_parameters_.ic_list
        sensor_height = trial.participant_metadata["sensor_height_m"]
        print(sensor_height)
        
        # Initialize the iterator
        iterator = GsIterator()

        # Iterate over gait sequences and calculate stride lengths
        for (gs, data), r in iterator.iterate(imu_data, reference_wbs):
            refined_gs, refined_ic_list = refine_gs(reference_ic.loc[gs.id])

            sl = sl_calculator.clone().calculate(
                data=to_body_frame(data),
                initial_contacts=refined_ic_list,
                sensor_height_m=sensor_height,
                sampling_rate_hz=sampling_rate_hz,
            )
            r.stride_length_per_sec = sl.stride_length_per_sec_

        # Store detected stride lengths and reference stride lengths
        all_detected_sl[trial.group_label] = iterator.results_.stride_length_per_sec.groupby("wb_id").mean()
        all_ref_sl[trial.group_label] = reference_wbs[["avg_stride_length_m"]].rename(
            columns={"avg_stride_length_m": "stride_length_m"}
        )

    # Concatenate results and return them
    index_names = ["cohort", "participant_id", "time_measure", "test", "trial"]
    detected_df = pd.concat(all_detected_sl, names=index_names)
    reference_df = pd.concat(all_ref_sl, names=index_names)
    
    return detected_df, reference_df

def process_all_participants(base_path):
    """Process all participants in the given base directory and return concatenated results."""
    all_detected = []
    all_reference = []
    
    # Iterate over all participant folders
    for participant_folder in os.listdir(base_path):
        participant_path = os.path.join(base_path, participant_folder)
        
        # Check if the folder contains the required data.mat file
        if os.path.isdir(participant_path) and "data.mat" in os.listdir(participant_path):
            print(f"Processing participant: {participant_folder}")
            
            # Process the participant data and get detected and reference stride lengths
            detected_sl, reference_sl = process_participant_data(participant_folder, participant_path)
            
            # Append results to the global list
            all_detected.append(detected_sl)
            all_reference.append(reference_sl)
        else:
            print(f"Skipping folder: {participant_folder}, no data.mat file found.")
    
    # Concatenate all participant results into a single DataFrame
    all_detected_combined = pd.concat(all_detected)
    all_reference_combined = pd.concat(all_reference)
    
    return all_detected_combined, all_reference_combined

# Define the base directory containing participant folders
#base_dir = 'C:/Users/ac4gt/Desktop/Mob-DPipeline/smartphone/test_data/lab/HA/'
base_dir = 'C:/PoliTO/Tesi/mobgap/smartphone/test_data/lab/HA/'

# Process all participants and get the concatenated results
all_detected_sl, all_ref_sl = process_all_participants(base_dir)

# Display concatenated results
display(all_detected_sl)
display(all_ref_sl)
# %% Error calculation

combined_sl = {
    "detected": all_detected_sl, 
    "reference": all_ref_sl
}
combined_sl = pd.concat(combined_sl, axis=1).reorder_levels((1, 0), axis=1)

errors = [
    ("stride_length_m", [E.error, E.abs_error, E.rel_error])
]

sl_errors = apply_transformations(combined_sl, errors)
combined_sl_with_errors = pd.concat([combined_sl, sl_errors], axis=1)

multiindex_column_order = [
    # 'stride_length_m' columns
    ('stride_length_m', 'detected'),
    ('stride_length_m', 'reference'),
    ('stride_length_m', 'error'),
    ('stride_length_m', 'abs_error'),
    ('stride_length_m', 'rel_error'),
]

combined_sl_with_errors = combined_sl_with_errors.reindex(columns=multiindex_column_order)
display(combined_sl_with_errors)
# %% Aggregate errors overall

aggregation = [
    *((("stride_length_m", o), ["mean", "std", A.quantiles]) for o in ["detected", "reference", "error", "abs_error", "rel_error"]),
    *(
        CustomOperation(identifier=m, function=A.icc, column_name=(m, "all"))
        for m in ["stride_length_m"]
    )
]

agg_results = (
    apply_aggregations(combined_sl_with_errors, aggregation)
    .rename_axis(index=["aggregation", "metric", "origin"])
    .reorder_levels(["metric", "origin", "aggregation"])
    .sort_index(level=0)
    .to_frame("values")
)

display(agg_results)

# %% Aggregate errors by participant

agg_by_participant = (
    combined_sl_with_errors.groupby("participant_id").apply(
        lambda df: apply_aggregations(df, aggregation)
    )
)

display(agg_by_participant.transpose())
# %% Save the results

# Create a results folder if it doesn't exist
results_folder = os.path.join(data_path, "results")
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

sl_with_errors_path = os.path.join(results_folder, f"stride_length_errors_subject_{subject_id}.csv")
combined_sl_with_errors.to_csv(sl_with_errors_path)

agg_results_path = os.path.join(results_folder, f"stride_length_agg_results_subject_{subject_id}.csv")
agg_results.to_csv(agg_results_path)

# %%

dataset = LabExampleDataset()

trial = dataset.get_subset("HA", "")