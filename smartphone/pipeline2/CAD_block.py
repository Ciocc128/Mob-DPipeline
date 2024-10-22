#%%
from mobgap.cadence import CadFromIc
from mobgap.data import GenericMobilisedDataset
from mobgap.pipeline import GsIterator
from mobgap.initial_contacts import refine_gs
from mobgap.utils.conversions import to_body_frame
import pandas as pd
from IPython.display import display
from mobgap.pipeline.evaluation import ErrorTransformFuncs as E
from mobgap.utils.df_operations import apply_transformations, apply_aggregations, CustomOperation
from mobgap.pipeline.evaluation import CustomErrorAggregations as A


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
# initialize the iterator
iterator = GsIterator()

# Initialize results storage for ic parameters and reference data
all_detected_cad = {}
all_ref_cad = {}

for trial in mobDataset[3:]:
    imu_data = trial.data_ss
    reference_ic = trial.reference_parameters_relative_to_wb_.ic_list
    reference_gs = trial.reference_parameters_relative_to_wb_.wb_list

    cad_from_ic = CadFromIc()

    for (gs, data), r in iterator.iterate(trial.data_ss, reference_gs):
        r.ic_list = reference_ic.loc[gs.id]
        refined_gs, refined_ic_list = refine_gs(r.ic_list)
        with iterator.subregion(refined_gs) as ((_, refined_gs_data), rr):
            cad = cad_from_ic.clone().calculate(
                to_body_frame(refined_gs_data),
                initial_contacts=refined_ic_list,
                sampling_rate_hz=trial.sampling_rate_hz,
            )
            rr.cadence_per_sec = cad.cadence_per_sec_

    
    all_detected_cad[trial.group_label] = iterator.results_.cadence_per_sec.groupby("wb_id").mean()
    all_ref_cad[trial.group_label] = reference_gs[["avg_cadence_spm"]].rename(
        columns={"avg_cadence_spm": "cadence_spm"}
    )

index_names = ["cohort", "participant_id", "time_measure", "test", "trial"]
all_detected_cad = pd.concat(all_detected_cad, names=index_names)
all_ref_cad = pd.concat(all_ref_cad, names=index_names)

display(all_detected_cad)
display(all_ref_cad)
# %% Error calculation
combined_cad = {"detected": all_detected_cad, "reference": all_ref_cad}
combined_cad = pd.concat(combined_cad, axis=1).reorder_levels((1, 0), axis=1)
display(combined_cad)

# error configuration
errors = [
    ("cadence_spm", [E.abs_error, E.rel_error])
]

cad_errors = apply_transformations(combined_cad, errors)
combined_cad_with_errors = pd.concat([combined_cad, cad_errors], axis=1)

display(combined_cad_with_errors)
# %% Aggregate errors

aggregation = [
    *[(("cadence_spm", o), ["mean", "std"]) for o in ["abs_error", "rel_error"]],
    CustomOperation(
        identifier="cadence_spm",
        function=A.icc,
        column_name=("cadence_spm", "all"),
    )
]

agg_results = (
    apply_aggregations(combined_cad_with_errors, aggregation)
    .rename_axis(index=["aggregation", "metric", "origin"])
    .reorder_levels(["metric", "origin", "aggregation"])
    .sort_index(level=0)
    .to_frame("values")
)

display(agg_results)
# %%
