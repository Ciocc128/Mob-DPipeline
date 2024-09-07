# %%
"""
This script processes gait data for a specified subject to detect initial contacts (ICs) using the Ionescu algorithm. 
The input data is assumed to be from free walking trials where each trial may contain multiple walking bouts. 

### Key Features:
- Loads data for a specific subject from a designated directory.
- Iterates over trials, detecting ICs using the Ionescu algorithm.
- Outputs results in a structured JSON format, including information about walking bouts.

### Data Processing Steps:
1. **Load Data**: The script loads gait data from a .mat file specific to the subject.
2. **IC Detection**: The Ionescu algorithm is applied to detect initial contacts (ICs) in the IMU data.
3. **Structure Output**: The ICs are stored in a nested JSON structure, including walking bout identifiers.

### Output Data Format:
The output is saved in a JSON file, structured as follows:

```json
{
  "ICD_Output": {
    "time_measure_1": {
      "test_1": {
        "trial_1": {
          "SU": {
            "LowerBack": {
              "ICD": [
                {
                  "wb_id": 1,   # Walking bout ID
                  "step_id": 1, # Step ID within the walking bout
                  "ic": 10      # Initial contact index in the data
                },
                {
                  "wb_id": 1,   # Same walking bout
                  "step_id": 2, # Next step ID
                  "ic": 20      # Next IC index
                },
                {
                  "wb_id": 2,   # Different walking bout
                  "step_id": 1, # Step ID for the new walking bout
                  "ic": 30      # IC index for the new bout
                },
                ...
              ]
            }
          }
        }
      }
    }
  }
}

"""
import json
import os 
import numpy as np
import matplotlib.pyplot as plt
from mobgap.data import GenericMobilisedDataset
from mobgap.icd import IcdIonescu
from mobgap.pipeline import GsIterator

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
data_output = {"ICD_Output": {}}

# Helper function to convert numpy types to native Python types
def convert_to_native_types(obj):
    if isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(elem) for elem in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj

# Process the trials from the 4th to the last one
for trial in mobDataset[3:]:
    short_trial = trial
    imu_data = short_trial.data_ss
    reference_wbs = short_trial.reference_parameters_.wb_list
    ref_ics = short_trial.reference_parameters_.ic_list
    iterator = GsIterator()

    for (gs, data), result in iterator.iterate(imu_data, reference_wbs):
        result.ic_list = (
            IcdIonescu().detect(data, sampling_rate_hz=short_trial.sampling_rate_hz).ic_list_
        )

    detected_ics = iterator.results_.ic_list

    # Extract time_measure, test, and trial values
    params = trial.get_params()
    subset_index = params['subset_index']
    time_measure = subset_index.iloc[0]['time_measure']
    test = subset_index.iloc[0]['test']
    trial_name = subset_index.iloc[0]['trial']

    # Ensure the nested dictionary structure exists
    if time_measure not in data_output["ICD_Output"]:
        data_output["ICD_Output"][time_measure] = {}

    if test not in data_output["ICD_Output"][time_measure]:
        data_output["ICD_Output"][time_measure][test] = {}

    if trial_name not in data_output["ICD_Output"][time_measure][test]:
        data_output["ICD_Output"][time_measure][test][trial_name] = {}

    if "SU" not in data_output["ICD_Output"][time_measure][test][trial_name]:
        data_output["ICD_Output"][time_measure][test][trial_name]["SU"] = {}

    if "LowerBack" not in data_output["ICD_Output"][time_measure][test][trial_name]["SU"]:
        data_output["ICD_Output"][time_measure][test][trial_name]["SU"]["LowerBack"] = {}

    if "ICD" not in data_output["ICD_Output"][time_measure][test][trial_name]["SU"]["LowerBack"]:
        data_output["ICD_Output"][time_measure][test][trial_name]["SU"]["LowerBack"]["ICD"] = []

    # Append the detected ICs to the list with validation
    for (wb_id, step_id), row in detected_ics.iterrows():
        ic_index = row["ic"]

        # Ensure wb_id, step_id, and ic_index are valid integers
        if not isinstance(wb_id, (int, np.integer)) or not isinstance(step_id, (int, np.integer)):
            print(f"Unexpected wb_id or step_id value: {wb_id}, {step_id}")
            continue  # Skip invalid entries

        if not isinstance(ic_index, (int, np.integer)) or not str(ic_index).isdigit():
            print(f"Unexpected ic_index value: {ic_index}")
            continue  # Skip invalid entries

        data_output["ICD_Output"][time_measure][test][trial_name]["SU"]["LowerBack"]["ICD"].append({
            "wb_id": int(wb_id),  # Ensure it's a native Python int
            "step_id": int(step_id),  # Ensure it's a native Python int
            "ic": int(ic_index)  # Ensure it's a native Python int
        })

        # Debugging print statements
        print(f"Appended to ICD field: wb_id={wb_id}, step_id={step_id}, ic={ic_index}")

    print("Reference Parameters:\n\n", ref_ics)
    print("\nPython Output:\n\n", detected_ics)

    print("Reference Parameters:\n\n", ref_ics)
    print("\nPython Output:\n\n", detected_ics)

    # Reset index for plotting
    imu_data.reset_index(drop=True).plot(y="acc_x")

    plt.plot(ref_ics["ic"], imu_data["acc_x"].iloc[ref_ics["ic"]], "o", label="ref")
    plt.plot(
        detected_ics["ic"],
        imu_data["acc_x"].iloc[detected_ics["ic"]],
        "x",
        label="icd_ionescu_py",
    )
    plt.legend()
    plt.show()


# Convert data_output to native Python types
data_output_native = convert_to_native_types(data_output)

""" # Determine the current script directory
current_directory = os.path.dirname(os.path.abspath(__file__))

 # Save the results to a JSON file in the current directory
json_file_path = os.path.join(current_directory, "icd_output.json")
with open(json_file_path, "w") as json_file:
    json.dump(data_output, json_file, indent=4)

print(f"Data saved to {json_file_path}") """


# Save the results to a JSON file
json_file_path = f"{data_path}icd_output.json"
with open(json_file_path, "w") as json_file:
    json.dump(data_output_native, json_file, indent=4)

print(f"Data saved to {json_file_path}")

# %%
