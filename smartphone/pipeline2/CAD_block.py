#%%
from mobgap.cadence import CadFromIcDetector
from mobgap.data import GenericMobilisedDataset
from mobgap.pipeline import GsIterator
from mobgap.initial_contacts import refine_gs, IcdShinImproved
from mobgap.utils.conversions import to_body_frame
import pandas as pd
import os
from IPython.display import display
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

def truncate_to_decimals(x):
    if isinstance(x, float):  # Applica solo ai numeri float
        return round(x, 3)
    elif isinstance(x, (tuple, list)):  # Se è un tuple o una lista, tronca ricorsivamente
        return type(x)(truncate_to_decimals(i) for i in x)
    return x  # Mantieni gli altri valori inalterati

#base_dir = 'C:/Users/ac4gt/Desktop/Mob-DPipeline/smartphone/test_data/lab/HA/'
base_dir = 'C:/PoliTO/Tesi/mobgap/smartphone/test_data/lab/HA/'

participant_folder = False#'011'
index_names = ["cohort", "participant_id", "time_measure", "test", "trial"]
participants = [participant_folder] if participant_folder else os.listdir(base_dir)

all_cad_with_errors = []

# initialize the iterator
iterator = GsIterator()
# Initialize the cadence detector
cad_from_ic = CadFromIcDetector(IcdShinImproved())

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

        # Initialize results storage for ic parameters and reference data
        all_detected_cad = {}
        all_ref_cad = {}

        for trial in mobDataset[3:]:
            imu_data = trial.data_ss
            reference_ic = trial.reference_parameters_relative_to_wb_.ic_list
            reference_gs = trial.reference_parameters_relative_to_wb_.wb_list

            cad_from_ic = CadFromIcDetector(IcdShinImproved())

            for (gs, data), r in iterator.iterate(trial.data_ss, reference_gs):
                r.ic_list = reference_ic.loc[gs.id]
                refined_gs, refined_ic_list = refine_gs(r.ic_list)
                with iterator.subregion(refined_gs) as ((_, refined_gs_data), rr):
                    cad = cad_from_ic.calculate(
                        to_body_frame(refined_gs_data),
                        initial_contacts=refined_ic_list,
                        sampling_rate_hz=trial.sampling_rate_hz,
                    )
                    rr.cadence_per_sec = cad.cadence_per_sec_

            
            all_detected_cad[trial.group_label] = iterator.results_.cadence_per_sec.groupby("wb_id").mean()
            all_ref_cad[trial.group_label] = reference_gs[["avg_cadence_spm"]].rename(
                columns={"avg_cadence_spm": "cadence_spm"}
            )

        all_detected_cad = pd.concat(all_detected_cad, names=index_names)
        all_ref_cad = pd.concat(all_ref_cad, names=index_names)

        combined_cad = {"detected": all_detected_cad, "reference": all_ref_cad}
        combined_cad = pd.concat(combined_cad, axis=1).reorder_levels((1, 0), axis=1)

        # error configuration
        errors = [
            ("cadence_spm", [E.abs_error, E.rel_error])
        ]

        cad_errors = apply_transformations(combined_cad, errors)
        combined_cad_with_errors = pd.concat([combined_cad, cad_errors], axis=1)
        all_cad_with_errors.append(combined_cad_with_errors)
        print('Cadence with errors:')
        display(combined_cad_with_errors)

        aggregation = [
            *[(("cadence_spm", o), ["mean", A.quantiles]) for o in ["abs_error", "detected"]],
            *[(("cadence_spm", o), ["mean", A.loa]) for o in ["rel_error"]],
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
        agg_results_trunc = agg_results.map(truncate_to_decimals)
        print('Aggregated results:')
        display(agg_results_trunc)

        # Save results for the participant
        save_results_to_folder(
            dataframes=[combined_cad_with_errors, agg_results],
            participant_folder=participant_path,
            filenames=[f"cad_with_errors_{participant_folder}.csv", f"cad_agg_results_{participant_folder}.csv"]
        )

        print('-----------------------------------')
        print('\n')

global_results = pd.concat(all_cad_with_errors)
print('Global cadence with errors:')
display(global_results)

# Calculate agg_results for the global_combined_tp_with_errors
global_aggregation = [
    *[(("cadence_spm", o), ["mean", A.quantiles]) for o in ["abs_error", "detected"]],
    *[(("cadence_spm", o), ["mean", A.loa]) for o in ["rel_error"]],
    *[CustomOperation(identifier="cadence_spm", function=A.icc, column_name=("cadence_spm", "all"))],
    CustomOperation(identifier=None, function=A.n_datapoints, column_name=("all", "all")),
]
global_agg_results = (
    apply_aggregations(global_results.dropna(), global_aggregation)
    .rename_axis(index=["aggregation", "metric", "origin"])
    .reorder_levels(["metric", "origin", "aggregation"])
    .sort_index(level=0)
    .to_frame("values")
)

print('Global aggregated results:')
global_agg_results_truncated = global_agg_results.map(truncate_to_decimals)
display(global_agg_results_truncated)

# Save the global results
results_path = os.path.join(base_dir, "CohortResults")
os.makedirs(results_path, exist_ok=True)
global_results.to_csv(os.path.join(results_path, "inLab_cad_with_errors.csv"))
global_agg_results.to_csv(os.path.join(results_path, "inLab_cad_agg_results.csv"))

print("All results saved successfully!")



# %%
