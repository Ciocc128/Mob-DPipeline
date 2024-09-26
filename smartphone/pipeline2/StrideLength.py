# %%
import json
import os
import numpy as np
from mobgap.data import GenericMobilisedDataset
from mobgap.stride_length import SlZijlstra
from mobgap.pipeline import GsIterator
from mobgap.initial_contacts import refine_gs
from mobgap.utils.conversions import to_body_frame
from smartphone.pipeline2.riorientation.riorientamento import process_and_rotate_dataset
import warnings
import matplotlib.pyplot as plt

"""
Stride Length Calculation Script using SlZijlstra

This script calculates the stride lengths for a specified subject using the SlZijlstra algorithm from the Mobgap library. 
The input data consists of IMU measurements recorded in a laboratory environment and processed to align with the global 
reference system.

### Features:
- **Data Loading**: Loads subject-specific gait data from a `.mat` file.
- **Reorientation Process**: IMU data is reoriented before stride length calculation using reference parameters.
- **Stride Length Calculation**: The SlZijlstra algorithm calculates stride lengths based on initial contacts (ICs) 
  detected from the reference system.
- **Refinement**: Gait sequences are refined using the `refine_gs` function to align with the first and last detected ICs.
- **Plotting**: Plots the vertical accelerometer data along with the reference gait sequence and initial contacts for 
  visual analysis.
- **JSON Output**: The final stride lengths are stored in a structured JSON format for each trial.

### Output Format:
The results are saved in a JSON file with the following structure:
```json
{
  "StrideLength_Output": {
    "time_measure_1": {
      "test_1": {
        "trial_1": {
          "SU": {
            "LowerBack": {
              "StrideLength": {
                "stride_length_per_sec": [1.10, 1.15, ...]  # Stride length values per second
              }
            }
          }
        }
      }
    }
  }
}
"""
# Suppress UserWarnings from the MobGap library
warnings.filterwarnings("ignore", category=UserWarning)

# Function to plot accelerometer data with reference WB and ICs
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

# Define subject and data paths
subject_id = "0005"
data_path = f'C:/Users/giorg/OneDrive - Politecnico di Torino/Giorgio Trentadue/Acquisizioni/Test/{subject_id}/In Lab/Results final/'

# Load the dataset
mobDataset = GenericMobilisedDataset(
    [os.path.join(data_path, "data.mat")],
    test_level_names=["time_measure", "test", "trial"],
    reference_system='INDIP',
    measurement_condition='laboratory',
    reference_para_level='wb',
    parent_folders_as_metadata=None
)

# Initialize output structure for stride length results
data_output = {"StrideLength_Output": {}}

# Helper function to convert numpy types to native Python types (for JSON serialization)
def convert_to_native_types(obj):
    if isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(elem) for elem in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj

# Iterate over trials in the dataset, starting from trial 3
for trial in mobDataset[3:]:
    # Extract metadata for the trial
    params = trial.get_params()
    subset_index = params['subset_index']
    time_measure = subset_index.iloc[0]['time_measure']
    test = subset_index.iloc[0]['test']
    trial_name = subset_index.iloc[0]['trial']

    # Extract trial data and parameters
    short_trial = trial
    imu_data = short_trial.data_ss
    reference_wbs = short_trial.reference_parameters_.wb_list
    ref_ics = short_trial.reference_parameters_.ic_list
    sampling_rate = short_trial.sampling_rate_hz
    sensor_height = short_trial.participant_metadata["sensor_height_m"]
    iterator = GsIterator()

    # Step 1: Reorient the IMU data
    reoriented_data = process_and_rotate_dataset(imu_data, exercise_name=f"{test} {trial_name}", visualize=True)

    # Step 2: Plot the reoriented data for debugging
    acc_x = reoriented_data['acc_x'].values
    plot_acc_with_wb_and_ics(acc_x, reference_wbs, ref_ics, f"{test} {trial_name}: Vertical Acceleration with WB and IC")

    # Step 3: Initialize the stride length calculator
    stride_length_calculator = SlZijlstra(
        **SlZijlstra.PredefinedParameters.step_length_scaling_factor_ms_ms
    )

    # Step 4: Iterate over gait sequences and refine them using initial contacts
    for (gs, data), result in iterator.iterate(reoriented_data, reference_wbs):
        refined_gs, refined_ic_list = refine_gs(ref_ics.loc[gs.id])

        print(f"\nRefined Gait Sequence: {refined_gs}")
        print(f"Length of Reoriented Data: {len(reoriented_data)}")
        print(f"Refined IC List: {refined_ic_list}")

        # Step 5: Calculate stride length for the refined gait sequence
        stride_length_result = stride_length_calculator.calculate(
            data=to_body_frame(data),
            initial_contacts=refined_ic_list,
            sensor_height_m=sensor_height,
            sampling_rate_hz=sampling_rate
        )

        # Output stride lengths per step
        sl_per_step = stride_length_result.raw_step_length_per_step_
        print(f"Stride Lengths per Step: {sl_per_step}")

        # Output stride lengths per second
        if hasattr(stride_length_result, 'stride_length_per_sec_'):
            print(f"Stride Length per Second: {stride_length_result.stride_length_per_sec_}")
        else:
            print("Stride length per second not found in the result")

        # Store the stride length per second in the result object
        result.stride_length_per_sec = stride_length_result.stride_length_per_sec_

    # Step 6: Extract detected stride lengths from iterator
    detected_stride_lengths = iterator.results_.stride_length_per_sec

    # Ensure the nested dictionary structure exists in the output
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

    # Collect stride lengths per second
    stride_lengths_per_sec = detected_stride_lengths["stride_length_m"].tolist()

    # Add stride lengths to the JSON structure
    data_output["StrideLength_Output"][time_measure][test][trial_name]["SU"]["LowerBack"]["StrideLength"] = {
        "stride_length_per_sec": stride_lengths_per_sec
    }

    print(f"\nStride Length Calculation Completed for {test} {trial_name}")
    print("Stride Lengths per Second:", stride_lengths_per_sec)
    print("\n\n\n")

# Convert output data to native types for JSON serialization
data_output_native = convert_to_native_types(data_output)

# Step 7: Save the output data to a JSON file in the results folder
script_directory = os.path.dirname(os.path.abspath(__file__))
result_directory = os.path.join(script_directory, 'results')

if not os.path.exists(result_directory):
    os.makedirs(result_directory)

json_file_path = os.path.join(result_directory, "stride_length_output.json")
with open(json_file_path, "w") as json_file:
    json.dump(data_output_native, json_file, indent=4)

print(f"Data saved to {json_file_path}")

# %%
