# %%
import json
import os
import numpy as np
from mobgap.data import GenericMobilisedDataset
from mobgap.stride_length import SlZijlstra
from mobgap.pipeline import GsIterator
from mobgap.utils.conversions import to_body_frame
from mobgap.initial_contacts import refine_gs
from smartphone.pipeline2.riorientamento import process_and_rotate_dataset  # Importing the reorientation function


"""

Stride Length Calculation Script

This script processes gait data for a specified subject to calculate stride lengths using the SlZijlstra algorithm from the Mobgap library.
The input data is assumed to be in the form of IMU measurements collected in a laboratory setting.

### Key Features:
- Loads data for a specific subject from a designated directory.
- Iterates over trials, calculating stride lengths using the SlZijlstra algorithm.
- Outputs results in a structured JSON format.

### Data Processing Steps:
1. **Load Data**: The script loads gait data from a .mat file specific to the subject.
2. **Stride Length Calculation**: The SlZijlstra algorithm is applied to calculate stride lengths based on initial contacts (ICs) detected in the previous pipeline stage.
3. **Structure Output**: The stride lengths are stored in a nested JSON structure, organized by time_measure, test, and trial.

### Output Data Format:
The output is saved in a JSON file, structured as follows:

```json
{
  "StrideLength_Output": {
    "time_measure_1": {
      "test_1": {
        "trial_1": {
          "SU": {
            "LowerBack": {
              "StrideLength": {
                "stride_length_per_sec": [1.105959, 1.160042, 1.088413, ...]  # List of stride lengths per second for the trial.
              }
            }
          }
        }
      }
    }
  }
}

"""

subject_id = "0005"
data_path = f'C:/Users/giorg/OneDrive - Politecnico di Torino/Giorgio Trentadue/Acquisizioni/Test/{subject_id}/In Lab/Results final/'

mobDataset = GenericMobilisedDataset( 
    [os.path.join(data_path, "data.mat")],
    test_level_names=["time_measure", "test", "trial"],
    reference_system='INDIP',
    measurement_condition='laboratory',
    reference_para_level='wb',
    parent_folders_as_metadata=None
)

data_output = {"StrideLength_Output": {}}  # Initialize output dictionary

def convert_to_native_types(obj): 
    if isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()} 
    elif isinstance(obj, list):
        return [convert_to_native_types(elem) for elem in obj]
    elif isinstance(obj, np.generic): 
        return obj.item() 
    else: 
        return obj

for trial in mobDataset[3:]:
    short_trial = trial 
    imu_data = short_trial.data_ss 
    reference_wbs = short_trial.reference_parameters_.wb_list 
    ref_ics = short_trial.reference_parameters_.ic_list 
    sampling_rate = short_trial.sampling_rate_hz 
    # infoForAlgo must be in the same directory as the folder "Result final", not in "Standarized" folder
    sensor_height = short_trial.participant_metadata["sensor_height_m"]
    iterator = GsIterator()

    # Initialize the stride length calculator with predefined parameters
    stride_length_calculator = SlZijlstra(
        **SlZijlstra.PredefinedParameters.step_length_scaling_factor_ms_ms
    )

    # Iterate over gait sequences and calculate stride length
    for (gs, data), result in iterator.iterate(imu_data, reference_wbs):

        # Align data to the body frame
        aligned_data = to_body_frame(data)

        # Calculate stride length
        stride_length_result = stride_length_calculator.calculate(
            data=aligned_data,
            initial_contacts=ref_ics,
            sensor_height_m=sensor_height,
            sampling_rate_hz=sampling_rate
        )

        if hasattr(stride_length_result, 'stride_length_per_sec_'):
            print(stride_length_result.stride_length_per_sec_)
        else:
            print("Stride length per second not found in the result")

        # Store the stride length per second
        result.stride_length_per_sec = stride_length_result.stride_length_per_sec_

    # Extract detected stride lengths
    detected_stride_lengths = iterator.results_.stride_length_per_sec
    print(detected_stride_lengths)

    # Extract time_measure, test, and trial values
    params = trial.get_params()
    subset_index = params['subset_index']
    time_measure = subset_index.iloc[0]['time_measure']
    test = subset_index.iloc[0]['test']
    trial_name = subset_index.iloc[0]['trial']

    # Ensure the nested dictionary structure exists
    if time_measure not in data_output["StrideLength_Output"]:
        data_output["StrideLength_Output"][time_measure] = {}

    if test not in data_output["StrideLength_Output"][time_measure]:
        data_output["StrideLength_Output"][time_measure][test] = {}

    if trial_name not in data_output["StrideLength_Output"][time_measure][test]:
        data_output["StrideLength_Output"][time_measure][test][trial_name] = {}

    if "SU" not in data_output["StrideLength_Output"][time_measure][test][trial_name]:
        data_output["StrideLength_Output"][time_measure][test][trial_name]["SU"] = {}

    if "LowerBack" not in data_output["StrideLength_Output"][time_measure][test][trial_name]["SU"]:
        data_output["StrideLength_Output"][time_measure][test][trial_name]["SU"]["LowerBack"] = {}

    if "StrideLength" not in data_output["StrideLength_Output"][time_measure][test][trial_name]["SU"]["LowerBack"]:
        data_output["StrideLength_Output"][time_measure][test][trial_name]["SU"]["LowerBack"]["StrideLength"] = {}

    # Collect stride lengths per second
    stride_lengths_per_sec = detected_stride_lengths["stride_length_m"].tolist()

    # Add stride lengths to JSON structure
    data_output["StrideLength_Output"][time_measure][test][trial_name]["SU"]["LowerBack"]["StrideLength"] = {
        "stride_length_per_sec": stride_lengths_per_sec
    }

    print("Stride Length Calculation Completed for Trial:")
    print(f"Time Measure: {time_measure}, Test: {test}, Trial: {trial_name}")
    print("Stride Lengths per Second:", stride_lengths_per_sec)

data_output_native = convert_to_native_types(data_output)

current_directory = os.path.dirname(os.path.abspath(__file__))

json_file_path = os.path.join(current_directory, "stride_length_output.json") 
with open(json_file_path, "w") as json_file:
    json.dump(data_output_native, json_file, indent=4)
print(f"Data saved to {json_file_path}")
# %%
