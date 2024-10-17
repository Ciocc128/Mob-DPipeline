from tqdm.auto import tqdm
from mobgap.data import LabExampleDataset, GenericMobilisedDataset
from mobgap.pipeline import MobilisedPipelineHealthy
from mobgap.initial_contacts import refine_gs
from mobgap.utils.conversions import to_body_frame
from smartphone.pipeline2.riorientation.riorientamento import process_and_rotate_dataset
import pandas as pd
import matplotlib.pyplot as plt

# Define subject ID and data path
subject_id = "005"
data_path = f'C:/Users/ac4gt/Desktop/Mob-DPipeline/smartphone/test_data/lab/HA/{subject_id}/'

# Load the dataset
mobDataset = GenericMobilisedDataset(
    [data_path + "data.mat"],
    test_level_names=["time_measure", "test", "trial"],
    parent_folders_as_metadata=["cohort", "participant_id"],
    reference_system='INDIP',
    measurement_condition='laboratory',
    reference_para_level='wb',
)

haPipeline = MobilisedPipelineHealthy()

per_wb_paras = {}
aggregated_paras = {}

# Iterate over trials in the dataset, starting from trial 3
for trial in tqdm(mobDataset[3:]):
    # Extract trial data and parameters
    params = trial.get_params()
    subset_index = params['subset_index']
    test = subset_index.iloc[0]['test']
    trial_name = subset_index.iloc[0]['trial']
    #trial.participant_metadata["cohort"] = "HA"
    print(trial.participant_metadata)
    print(trial.group_label)
    imu_data = trial.data_ss
    reference_wbs = trial.reference_parameters_.wb_list

    # reorienting data
    """"reoriented_data = process_and_rotate_dataset(imu_data, exercise_name=f"{test} {trial_name}", visualize=False)
    reoriented_trial = trial
    reoriented_trial.data_ss = reoriented_data"""

    pipe = haPipeline.clone().safe_run(trial)
    if not (per_wb := pipe.per_wb_parameters_).empty:
        per_wb_paras[trial.group_label] = per_wb
    if not (agg := pipe.aggregated_parameters_).empty:
        aggregated_paras[trial.group_label] = agg

per_wb_paras = pd.concat(per_wb_paras)
aggregated_paras = (
    pd.concat(aggregated_paras)
    .reset_index(-1, drop=True)
    .rename_axis(LabExampleDataset().index.columns)
    .reindex(pd.MultiIndex.from_tuples(LabExampleDataset().group_labels))
)

# Tabular report
print("Per Walking Bout Parameters:")
print(per_wb_paras)

print("\nAggregated Parameters:")
print(aggregated_paras)

# Plot aggregated parameters
aggregated_paras.plot(kind='bar', figsize=(10,6))
plt.title("Aggregated Parameters from Pipeline")
plt.xlabel("Groups")
plt.ylabel("Values")
plt.show()
