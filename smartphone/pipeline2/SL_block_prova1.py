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

base_dir = 'C:/Users/ac4gt/Desktop/Mob-DPipeline/smartphone/test_data/lab/HA/'
participant_folder = False
#participant_folder = '011'
index_names = ["cohort", "participant_id", "time_measure", "test", "trial"]
participants = [participant_folder] if participant_folder else os.listdir(base_dir)

all_sl_with_errors = []
# Initialize the iterator
iterator = GsIterator()

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
            #print(sensor_height)

            # Iterate over gait sequences and calculate stride lengths
            for (gs, data), r in iterator.iterate(imu_data, reference_wbs):
                refined_gs, refined_ic_list = refine_gs(reference_ic.loc[gs.id])

                sl = sl_calculator.calculate(
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
        detected_df = pd.concat(all_detected_sl, names=index_names)
        reference_df = pd.concat(all_ref_sl, names=index_names)

        combined_sl = {
            "detected": detected_df, 
            "reference": reference_df
        }
        combined_sl = pd.concat(combined_sl, axis=1).reorder_levels((1, 0), axis=1)
        errors = [
            ("stride_length_m", [E.abs_error, E.rel_error])
        ]

        sl_errors = apply_transformations(combined_sl, errors)
        combined_sl_with_errors = pd.concat([combined_sl, sl_errors], axis=1)

        multiindex_column_order = [
            # 'stride_length_m' columns
            ('stride_length_m', 'detected'),
            ('stride_length_m', 'reference'),
            ('stride_length_m', 'abs_error'),
            ('stride_length_m', 'rel_error'),
        ]

        combined_sl_with_errors = combined_sl_with_errors.reindex(columns=multiindex_column_order)
        all_sl_with_errors.append(combined_sl_with_errors)
        display(combined_sl_with_errors)

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

        # Save results for the participant
        save_results_to_folder(
            dataframes=[combined_sl_with_errors, agg_results],
            participant_folder=participant_path,
            filenames=[f"sl_with_errors_{participant_folder}.csv", f"sl_agg_results_{participant_folder}.csv"]
        )

global_results = pd.concat(all_sl_with_errors)

# Calculate agg_results for the global_combined_tp_with_errors
global_aggregation = [
    *[(("stride_length_m", o), ["mean", "std"]) for o in ["abs_error", "rel_error"]],
    CustomOperation(identifier="stride_length_m", function=A.icc, column_name=("stride_length_m", "all"))
]
global_agg_results = (
    apply_aggregations(global_results.dropna(), global_aggregation)
    .rename_axis(index=["aggregation", "metric", "origin"])
    .reorder_levels(["metric", "origin", "aggregation"])
    .sort_index(level=0)
    .to_frame("values")
)

agg_by_participant = (
    global_results.groupby("participant_id").apply(
        lambda df: apply_aggregations(df, aggregation)
    )
)
display(agg_by_participant.transpose())

# Save the global results
results_path = os.path.join(base_dir, "CohortResults")
os.makedirs(results_path, exist_ok=True)
global_results.to_csv(os.path.join(results_path, "inLab_sl_with_errors.csv"))
global_agg_results.to_csv(os.path.join(results_path, "inLab_sl_agg_results.csv"))

print("All results saved successfully!")
# %%
