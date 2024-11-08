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



def plot_acc_with_wb_and_ics(acc_x, reference_wbs, ref_ics, title="Accelerometer Signal with Reference WB and ICs"):
    plt.figure(figsize=(12, 6))
    plt.plot(acc_x, label="Vertical Acc (acc_x)", color='blue')

    # Plot reference WB as green shaded areas
    for idx, row in reference_wbs.iterrows():
        plt.axvspan(row['start'], row['end'], color='green', alpha=0.3, label="Reference WB" if idx == 0 else "")

    # Plot the reference ICs as black circles
    plt.scatter(ref_ics['ic'], acc_x[ref_ics['ic']], color='black', marker='o', label="Reference IC", zorder=5)

    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel("Accelerometer Value")
    plt.legend()
    plt.show()

# Define subject ID and data path
subject_id = "003"
data_path = f'C:/Users/ac4gt/Desktop/Mob-DPipeline/smartphone/test_data/lab/HA/{subject_id}/'

# Load the dataset
mobDataset = GenericMobilisedDataset(
    [data_path + "data.mat"],
    test_level_names=["time_measure", "test", "trial"],
    reference_system='INDIP',
    measurement_condition='laboratory',
    reference_para_level='wb',
    parent_folders_as_metadata=["cohort", "participant_id"]
)

all_detected_sl = {}
all_ref_sl = {}

sl_calculator = SlZijlstra(
    **SlZijlstra.PredefinedParameters.step_length_scaling_factor_ms_ms

)

for trial in mobDataset[3:]:
    imu_data = trial.data_ss
    sampling_rate_hz = trial.sampling_rate_hz
    reference_wbs = trial.reference_parameters_.wb_list
    reference_ic = trial.reference_parameters_.ic_list
    sensor_height = trial.participant_metadata["sensor_height_m"]
    display(trial.participant_metadata_as_df)

    # initialize the iterator
    iterator = GsIterator()

    # Check if the iterator yields anything
    for (gs, data), r in iterator.iterate(imu_data, reference_wbs):
        refined_gs, refined_ic_list = refine_gs(reference_ic.loc[gs.id])

        sl = sl_calculator.clone().calculate(
            data=to_body_frame(data),
            initial_contacts=refined_ic_list,
            sensor_height_m=sensor_height,
            sampling_rate_hz=sampling_rate_hz,
        )
        r.stride_length_per_sec = sl.stride_length_per_sec_

    all_detected_sl[trial.group_label] = iterator.results_.stride_length_per_sec.groupby("wb_id").mean()
    all_ref_sl[trial.group_label] = reference_wbs[["avg_stride_length_m"]].rename(
        columns= {"avg_stride_length_m": "stride_length_m"}
    )


index_names = ["cohort", "participant_id", "time_measure", "test", "trial"]
all_detected_sl = pd.concat(all_detected_sl, names=index_names)
all_ref_sl = pd.concat(all_ref_sl, names=index_names)

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

# %% Aggregate errors

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
# %% Save the results

# Create a results folder if it doesn't exist
results_folder = os.path.join(data_path, "results")
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

sl_with_errors_path = os.path.join(results_folder, f"stride_length_errors_subject_{subject_id}.csv")
combined_sl_with_errors.to_csv(sl_with_errors_path)

agg_results_path = os.path.join(results_folder, f"stride_length_agg_results_subject_{subject_id}.csv")
agg_results.to_csv(agg_results_path)