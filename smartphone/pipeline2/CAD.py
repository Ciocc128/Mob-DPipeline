# %%
import json
import os
import numpy as np
import pandas as pd
from mobgap.data import GenericMobilisedDataset
from mobgap.cad import CadFromIc
import math  

# Define subject ID and data path
subject_id = "005"
data_path = f'C:/Users/giorg/OneDrive - Politecnico di Torino/Giorgio Trentadue/Acquisizioni/{subject_id}/In Lab/Results final/'

# Load the dataset
mobDataset = GenericMobilisedDataset(
    [data_path + "data.mat"],
    test_level_names=["time_measure", "test", "trial"],
    reference_system='INDIP',
    measurement_condition='laboratory',
    reference_para_level='wb',
    parent_folders_as_metadata=None
)

# Initialize the JSON structure
data_output = {"CAD_Output": {}}

# Function to truncate a float to three decimal places
def truncate_float(value, decimal_places):
    factor = 10.0 ** decimal_places
    return math.trunc(value * factor) / factor

# Process the trials from the 4th to the last one
for trial in mobDataset[3:]:
    short_trial = trial
    imu_data = short_trial.data_ss
    reference_ic = short_trial.reference_parameters_relative_to_wb_.ic_list
    reference_gs = short_trial.reference_parameters_relative_to_wb_.wb_list

    cad_from_ic = CadFromIc()

    gs_id = reference_gs.index[0]
    data_in_gs = short_trial.data_ss.iloc[
        reference_gs.start.iloc[0]:reference_gs.end.iloc[0]
    ]
    ics_in_gs = reference_ic[["ic"]].loc[gs_id]

    cad_from_ic.calculate(
        data_in_gs, ics_in_gs, sampling_rate_hz=short_trial.sampling_rate_hz
    )

    cadence_df = cad_from_ic.cadence_per_sec_

    # Extract time_measure, test, and trial values
    params = trial.get_params()
    subset_index = params['subset_index']
    time_measure = subset_index.iloc[0]['time_measure']
    test = subset_index.iloc[0]['test']
    trial_name = subset_index.iloc[0]['trial']

    # Ensure the nested dictionary structure exists
    if time_measure not in data_output["CAD_Output"]:
        data_output["CAD_Output"][time_measure] = {}

    if test not in data_output["CAD_Output"][time_measure]:
        data_output["CAD_Output"][time_measure][test] = {}

    if trial_name not in data_output["CAD_Output"][time_measure][test]:
        data_output["CAD_Output"][time_measure][test][trial_name] = {}

    if "SU" not in data_output["CAD_Output"][time_measure][test][trial_name]:
        data_output["CAD_Output"][time_measure][test][trial_name]["SU"] = {}

    if "LowerBack" not in data_output["CAD_Output"][time_measure][test][trial_name]["SU"]:
        data_output["CAD_Output"][time_measure][test][trial_name]["SU"]["LowerBack"] = {}

    if "CAD" not in data_output["CAD_Output"][time_measure][test][trial_name]["SU"]["LowerBack"]:
        data_output["CAD_Output"][time_measure][test][trial_name]["SU"]["LowerBack"]["CAD"] = {}

    # Store cadence data
    sec_center_samples = cadence_df.index.tolist()

    # Truncate cadence_spm to three decimal places
    cadence_spm = [truncate_float(cad, 3) for cad in cadence_df["cadence_spm"].tolist()]

    # Add cadence data to JSON structure
    data_output["CAD_Output"][time_measure][test][trial_name]["SU"]["LowerBack"]["CAD"] = {
        "sec_center_sample": sec_center_samples,
        "cadence_spm": cadence_spm
    }

 # Determine the current script directory
current_directory = os.path.dirname(os.path.abspath(__file__))

 # Save the results to a JSON file in the current directory
json_file_path = os.path.join(current_directory, "cad_output.json")
with open(json_file_path, "w") as json_file:
    json.dump(data_output, json_file, indent=4)

print(f"Data saved to {json_file_path}")

""" # Output the data to a JSON file
output_path = f'{data_path}cad_output.json'
with open(output_path, 'w') as json_file:
    json.dump(data_output, json_file, indent=4)


print(f"CAD data successfully saved to {output_path}") """

# %%
